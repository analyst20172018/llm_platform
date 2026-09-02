"""Archon: an inference-time architecture that pipelines several LLM calls.

Based on "Archon: An Architecture Search Framework for Inference-Time
Techniques" (Saad-Falcon et al.). Instead of sending a prompt to one model,
the prompt flows through a pipeline of LLM components that generate,
critique, filter, and merge candidate answers:

    Generators (parallel) -> Critic -> Ranker -> Verifier -> Unit Tests -> Fuser

Every component except the Generators and the final Fuser is optional and is
enabled simply by supplying a model name for it. All LLM calls are made
through the platform's ``APIHandler``, so any model from ``models_config.yaml``
can play any role, and image (or other file) attachments are supported.

Usage:
    from llm_platform.self_improving_agents.archon import Archon

    archon = Archon(
        generator_models=["gpt-5.2", "claude-4.5-opus", "gemini-3-pro"],
        critic_model="claude-4.5-opus",
        ranker_model="gpt-5.2",
        fuser_model="claude-4.5-opus",
    )
    
    result = archon.generate(
        "Explain the Monty Hall problem", 
        files=[image],
        generator_parameters={"web_search": True},
    )
    print(result.answer)

``generate`` is synchronous; use ``await archon.generate_async(...)`` from
inside an already-running event loop.
"""

import ast
import asyncio
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from loguru import logger

from llm_platform.core.llm_handler import APIHandler
from llm_platform.services.files import BaseFile

# ---------------------------------------------------------------------------
# Component prompts (Tables 10-23 of the Archon paper)
# ---------------------------------------------------------------------------

# Table 10: the Generator receives the instruction itself, unchanged.

# Table 11
FUSER_PROMPT_WITH_CRITIQUES = """\
You have been provided with a set of responses with their individual critiques of \
strengths/weaknesses from various models to the latest user query. Your task is to \
synthesize these responses into a single, high-quality response. It is crucial to \
critically evaluate the information provided in these responses and their provided \
critiques of strengths/weaknesses, recognizing that some of it may be biased or \
incorrect. Your response should not simply replicate the given answers but should \
offer a refined, accurate, and comprehensive reply to the instruction. Ensure your \
response is well-structured, coherent, and adheres to the highest standards of \
accuracy and reliability.

Responses from models:
{responses}

{instruction}"""

# Table 12
FUSER_PROMPT_WITHOUT_CRITIQUES = """\
You have been provided with a set of responses from various models to the latest \
user query. Your task is to synthesize these responses into a single, high-quality \
response. It is crucial to critically evaluate the information provided in these \
responses, recognizing that some of it may be biased or incorrect. Your response \
should not simply replicate the given answers but should offer a refined, accurate, \
and comprehensive reply to the instruction. Ensure your response is well-structured, \
coherent, and adheres to the highest standards of accuracy and reliability.

{responses}

{instruction}"""

# Table 14
RANKER_PROMPT = """\
I will provide you with {n} responses, each indicated by a numerical identifier []. \
Rank the responses based on their relevance to the instruction: {instruction}

{responses}

Instruction: {instruction}

Rank the {n} responses above based on their relevance to the instruction. All the \
responses should be included and listed using identifiers, in descending order of \
relevance to the instruction. The output format should be [] > [], e.g., [4] > [2]. \
Only respond with the ranking results, do not say any word or explain."""

# Table 15
CRITIC_PROMPT = """\
You are a helpful assistant. I will provide you with {n} responses, each indicated \
by a numerical identifier (e.g., [1], [2], etc.). Evaluate the responses based on \
their relevance to the instruction: {instruction}

{responses}

Instruction: {instruction}

Evaluate the {n} responses above based on their relevance to the instruction. All \
the responses should be included and listed using identifiers. For each response, \
start the critique with the numerical identifier (e.g., [1]) followed by the \
strengths and weaknesses. You must include both strengths and weaknesses, even if \
there are more of one than the other. At the end of each response's analysis, \
include two new lines to separate the critiques. Do not include any preface or text \
after the critiques. Do not include any references to previous critiques within a \
critique. Start with the analysis for the first response and end with the analysis \
for the last response. Structure each response's analysis as follows:
Strengths:
- <strength #1>
- <strength #2>
Weaknesses:
- <weakness #1>
- <weakness #2>"""

# Table 16 (Verifier stage 1: produce reasoning)
VERIFIER_REASONING_PROMPT = """\
I will provide you with a response indicated by the identifier 'Response'. Provide \
reasoning for why the response accurately and completely addresses the instruction: \
{instruction}

Response: {response}

Instruction: {instruction}

Provide the reasoning for the response above based on its relevance, completeness, \
and accuracy when compared to the instruction. Do not include any preface or text \
after the reasoning."""

# Verifier stage 2 (verdict), per the two-stage procedure described in Section 3.1
VERIFIER_VERDICT_PROMPT = """\
I will provide you with an instruction, a candidate response, and reasoning about \
that response. Based on the instruction, the response, and the reasoning, decide \
whether the candidate response correctly and completely addresses the instruction.

Instruction: {instruction}

Candidate Response: {response}

Reasoning: {reasoning}

Briefly explain your decision, then finish with a single verdict on its own line, \
formatted exactly as '[Correct]' or '[Incorrect]'."""

# Table 17
UNIT_TEST_GENERATOR_PROMPT = """\
Given the following query, generate a set of {n} unit tests that would evaluate the \
correctness of responses to this query.
- The unit tests should cover various aspects of the query and ensure comprehensive evaluation.
- Each unit test should be clearly stated and should include the expected outcome.
- The unit tests should be in the form of assertions that can be used to validate the correctness of responses to the query.
- The unit test should be formatted like 'The answer mentions...', 'The answer states...', 'The answer uses...', etc. followed by the expected outcome.
- Solely provide the unit tests for the question below. Do not provide any text before or after the list. Only output the unit tests as a list of strings (e.g., ['unit test #1', 'unit test #2', 'unit test #3']).

Query: {instruction}"""

# Table 23
UNIT_TEST_EVALUATOR_PROMPT = """\
Given the following query, candidate response, and unit tests, evaluate whether or \
not the response passes each unit test.
- In your evaluation, you should consider how the response aligns with the unit tests and query.
- Provide reasoning before you return your evaluation.
- At the end of your evaluation, you must finish with a list of verdicts corresponding to each unit test.
- You must include a verdict with one of these formatted options: '[Passed]' or '[Failed]'.
- Here is an example of the output format:
Unit Test #1: [Passed]
Unit Test #2: [Failed]
Unit Test #3: [Passed]
- Each verdict should be on a new line and correspond to the unit test in the same position.
- Here is the query, response, and unit tests for your evaluation:

Query: {instruction}

Candidate Response: {response}

Unit Tests:
{unit_tests}"""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """One candidate answer and everything the pipeline learned about it."""
    text: str
    model: str
    critique: Optional[str] = None
    rank: Optional[int] = None            # 1 = best, assigned by the ranker
    verified: Optional[bool] = None
    tests_passed: Optional[int] = None
    dropped_by: Optional[str] = None      # "ranker" | "verifier" | "unit_tests"

    @property
    def active(self) -> bool:
        return self.dropped_by is None


@dataclass
class ArchonResult:
    """Final answer plus the full trace of intermediate pipeline state."""
    answer: str
    candidates: List[Candidate]
    unit_tests: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

class Archon:
    """A configurable Archon inference-time pipeline.

    Args:
        generator_models: Model names producing the initial candidate answers.
            The same name may appear several times; ``samples_per_generator``
            additionally repeats every model that many times.
        fuser_model: Model that merges the surviving candidates into the final
            answer (the last layer is always a single fuser).
        critic_model: Optional model listing strengths/weaknesses per candidate.
        ranker_model: Optional model ordering candidates; only the ``top_k``
            best are kept.
        verifier_model: Optional model running the two-stage verification;
            candidates judged incorrect are dropped.
        unit_test_model: Optional model that writes ``num_unit_tests``
            assertions from the prompt and scores each candidate against them;
            only the best-scoring candidates are kept.
        samples_per_generator: How many times each generator model is sampled.
        top_k: How many candidates the ranker keeps.
        num_unit_tests: How many unit tests to generate.
        system_prompt: System prompt used for every LLM call.
    """

    def __init__(
        self,
        generator_models: List[str],
        fuser_model: str,
        critic_model: Optional[str] = None,
        ranker_model: Optional[str] = None,
        verifier_model: Optional[str] = None,
        unit_test_model: Optional[str] = None,
        samples_per_generator: int = 1,
        top_k: int = 3,
        num_unit_tests: int = 5,
        system_prompt: str = "You are a helpful assistant",
    ):
        if not generator_models:
            raise ValueError("At least one generator model is required")
        if not fuser_model:
            raise ValueError("A fuser model is required (the last layer is always a fuser)")

        self.generator_models = generator_models
        self.fuser_model = fuser_model
        self.critic_model = critic_model
        self.ranker_model = ranker_model
        self.verifier_model = verifier_model
        self.unit_test_model = unit_test_model
        self.samples_per_generator = samples_per_generator
        self.top_k = top_k
        self.num_unit_tests = num_unit_tests
        self.system_prompt = system_prompt

    # ------------------------------------------------------------- public API

    def generate(
        self,
        prompt: str,
        files: Optional[List[BaseFile]] = None,
        generator_parameters: Optional[Dict] = None,
    ) -> ArchonResult:
        """Run the pipeline synchronously. See :meth:`generate_async`."""
        return asyncio.run(self.generate_async(prompt, files, generator_parameters))

    async def generate_async(
        self,
        prompt: str,
        files: Optional[List[BaseFile]] = None,
        generator_parameters: Optional[Dict] = None,
    ) -> ArchonResult:
        """Run the pipeline: generate candidates, refine them, fuse the answer.

        Args:
            prompt: The user instruction.
            files: Optional attachments (e.g. ``ImageFile``); they are passed
                to every component so critics/rankers/fusers can judge the
                candidates against them.
            generator_parameters: Optional ``additional_parameters`` applied to
                the generator calls only (e.g. reasoning effort, temperature).

        Returns:
            ArchonResult with the fused answer and the per-candidate trace.
        """
        candidates = await self._generate(prompt, files, generator_parameters)

        if self.critic_model:
            await self._critique(prompt, candidates, files)
        if self.ranker_model:
            await self._rank(prompt, candidates, files)
        if self.verifier_model:
            await self._verify(prompt, candidates, files)

        unit_tests = None
        if self.unit_test_model:
            unit_tests = await self._run_unit_tests(prompt, candidates, files)

        answer = await self._fuse(prompt, candidates, files)
        return ArchonResult(answer=answer, candidates=candidates, unit_tests=unit_tests)

    # ---------------------------------------------------------------- helpers

    async def _call(
        self,
        model: str,
        prompt: str,
        files: Optional[List[BaseFile]] = None,
        additional_parameters: Optional[Dict] = None,
    ) -> str:
        """One isolated LLM call through a fresh APIHandler conversation."""
        handler = APIHandler(system_prompt=self.system_prompt)
        message = await handler.request_async(
            model,
            prompt,
            files=files,
            additional_parameters=additional_parameters,
        )
        return message.content or ""

    @staticmethod
    def _active(candidates: List[Candidate]) -> List[Candidate]:
        return [c for c in candidates if c.active]

    @staticmethod
    def _numbered_responses(candidates: List[Candidate], brackets: bool = False) -> str:
        """Render candidates as '1. <text>' (or '[1] <text>') blocks."""
        template = "[{i}] {text}" if brackets else "{i}. {text}"
        return "\n\n".join(
            template.format(i=i, text=c.text) for i, c in enumerate(candidates, start=1)
        )

    # ------------------------------------------------------------- components

    async def _generate(
        self,
        prompt: str,
        files: Optional[List[BaseFile]],
        generator_parameters: Optional[Dict],
    ) -> List[Candidate]:
        """First layer: sample every generator model in parallel."""
        models = [m for m in self.generator_models for _ in range(self.samples_per_generator)]
        logger.info(f"Archon: generating {len(models)} candidates")

        results = await asyncio.gather(
            *(self._call(m, prompt, files, generator_parameters) for m in models),
            return_exceptions=True,
        )

        candidates = []
        for model, result in zip(models, results):
            if isinstance(result, BaseException):
                logger.warning(f"Archon: generator '{model}' failed: {result}")
            elif result.strip():
                candidates.append(Candidate(text=result, model=model))

        if not candidates:
            raise RuntimeError("All Archon generators failed to produce a response")
        return candidates

    async def _critique(
        self, prompt: str, candidates: List[Candidate], files: Optional[List[BaseFile]]
    ) -> None:
        """Critic layer: one call producing strengths/weaknesses per candidate."""
        active = self._active(candidates)
        logger.info(f"Archon: critiquing {len(active)} candidates with {self.critic_model}")

        critic_prompt = CRITIC_PROMPT.format(
            n=len(active),
            instruction=prompt,
            responses=self._numbered_responses(active, brackets=True),
        )
        output = await self._call(self.critic_model, critic_prompt, files)

        for i, candidate in enumerate(active, start=1):
            candidate.critique = self._extract_critique(output, i)

    @staticmethod
    def _extract_critique(critic_output: str, index: int) -> str:
        """Pull the '[index] ...' section out of the critic's combined output."""
        matches = list(re.finditer(r"^\s*\[(\d+)\]", critic_output, flags=re.MULTILINE))
        for pos, match in enumerate(matches):
            if int(match.group(1)) == index:
                end = matches[pos + 1].start() if pos + 1 < len(matches) else len(critic_output)
                return critic_output[match.end():end].strip()
        # Parsing failed for this index: give the fuser the whole critic output
        return critic_output.strip()

    async def _rank(
        self, prompt: str, candidates: List[Candidate], files: Optional[List[BaseFile]]
    ) -> None:
        """Ranker layer: order the candidates, keep only the top-K."""
        active = self._active(candidates)
        if len(active) <= 1:
            return
        logger.info(f"Archon: ranking {len(active)} candidates with {self.ranker_model}")

        ranker_prompt = RANKER_PROMPT.format(
            n=len(active),
            instruction=prompt,
            responses=self._numbered_responses(active, brackets=True),
        )
        output = await self._call(self.ranker_model, ranker_prompt, files)

        # Parse identifiers like "[4] > [2] > [1]"; append any the model omitted
        order = []
        for match in re.findall(r"\[(\d+)\]", output):
            index = int(match) - 1
            if 0 <= index < len(active) and index not in order:
                order.append(index)
        order += [i for i in range(len(active)) if i not in order]

        for rank, index in enumerate(order, start=1):
            active[index].rank = rank
            if rank > self.top_k:
                active[index].dropped_by = "ranker"

    async def _verify(
        self, prompt: str, candidates: List[Candidate], files: Optional[List[BaseFile]]
    ) -> None:
        """Verifier layer: two-stage reasoning + verdict for every candidate."""
        active = self._active(candidates)
        logger.info(f"Archon: verifying {len(active)} candidates with {self.verifier_model}")

        async def verify_one(candidate: Candidate) -> bool:
            reasoning = await self._call(
                self.verifier_model,
                VERIFIER_REASONING_PROMPT.format(instruction=prompt, response=candidate.text),
                files,
            )
            verdict = await self._call(
                self.verifier_model,
                VERIFIER_VERDICT_PROMPT.format(
                    instruction=prompt, response=candidate.text, reasoning=reasoning
                ),
                files,
            )
            # Fail open: an unparseable verdict does not drop the candidate
            return "[Incorrect]" not in verdict

        verdicts = await asyncio.gather(*(verify_one(c) for c in active))
        for candidate, verified in zip(active, verdicts):
            candidate.verified = verified

        if any(verdicts):
            for candidate, verified in zip(active, verdicts):
                if not verified:
                    candidate.dropped_by = "verifier"
        else:
            logger.warning("Archon: verifier rejected every candidate; keeping all")

    async def _run_unit_tests(
        self, prompt: str, candidates: List[Candidate], files: Optional[List[BaseFile]]
    ) -> List[str]:
        """Unit test layers: generate assertions, score candidates, keep the best."""
        logger.info(f"Archon: generating unit tests with {self.unit_test_model}")
        output = await self._call(
            self.unit_test_model,
            UNIT_TEST_GENERATOR_PROMPT.format(n=self.num_unit_tests, instruction=prompt),
            files,
        )
        unit_tests = self._parse_unit_tests(output)
        if not unit_tests:
            logger.warning("Archon: could not parse any unit tests; skipping evaluation")
            return []

        active = self._active(candidates)
        tests_block = "\n".join(
            f"Unit Test #{i}: {test}" for i, test in enumerate(unit_tests, start=1)
        )

        async def evaluate_one(candidate: Candidate) -> int:
            evaluation = await self._call(
                self.unit_test_model,
                UNIT_TEST_EVALUATOR_PROMPT.format(
                    instruction=prompt, response=candidate.text, unit_tests=tests_block
                ),
                files,
            )
            return len(re.findall(r"\[Passed\]", evaluation))

        scores = await asyncio.gather(*(evaluate_one(c) for c in active))
        for candidate, score in zip(active, scores):
            candidate.tests_passed = score

        best = max(scores)
        for candidate, score in zip(active, scores):
            if score < best:
                candidate.dropped_by = "unit_tests"
        return unit_tests

    @staticmethod
    def _parse_unit_tests(output: str) -> List[str]:
        """Parse the generator's list-of-strings output, tolerating extra text."""
        match = re.search(r"\[.*\]", output, flags=re.DOTALL)
        if match:
            try:
                parsed = ast.literal_eval(match.group(0))
                if isinstance(parsed, list):
                    return [str(test) for test in parsed if str(test).strip()]
            except (ValueError, SyntaxError):
                pass
        # Fallback: treat every non-empty line as one test
        lines = [line.strip(" -*") for line in output.splitlines()]
        return [line for line in lines if line]

    async def _fuse(
        self, prompt: str, candidates: List[Candidate], files: Optional[List[BaseFile]]
    ) -> str:
        """Last layer: a single fuser merges the surviving candidates."""
        active = self._active(candidates)
        logger.info(f"Archon: fusing {len(active)} candidates with {self.fuser_model}")

        with_critiques = all(c.critique for c in active)
        if with_critiques:
            responses = "\n\n".join(
                f"{i}. {c.text}\nCritique: {c.critique}"
                for i, c in enumerate(active, start=1)
            )
            fuser_prompt = FUSER_PROMPT_WITH_CRITIQUES.format(
                responses=responses, instruction=prompt
            )
        else:
            fuser_prompt = FUSER_PROMPT_WITHOUT_CRITIQUES.format(
                responses=self._numbered_responses(active), instruction=prompt
            )
        return await self._call(self.fuser_model, fuser_prompt, files)
