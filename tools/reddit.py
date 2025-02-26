import praw
from llm_platform.tools.base import BaseTool
from typing import Dict
from pydantic import BaseModel, Field

class Reddit(BaseTool):
    """Reads posts and comments from the given subreddit in reddit.
    """

    __name__ = "Reddit"

    class InputModel(BaseModel):
        subreddit: str = Field(description = "Name of subreddit to read without symbols r/ .", required = True)
        include_comments: bool = Field(description = "Should comments be included.", required = True)

    def __init__(self, client_id, client_secret, user_agent):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.client = None

    def __call__(self, subreddit: str, include_comments: bool) -> Dict:
        """Reads posts and comments from the given subreddit in reddit."""
        # Initialize Reddit client if not already initialized
        if self.client is None:
            self.client = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent
            )


        subreddit = self.client.subreddit(subreddit)

        outputs = []
        for submission in subreddit.new(limit=120):
            output_item = {
                "title": submission.title,
                "selftext": submission.selftext,
                "comments": []
            }
            
            # Add comments
            if include_comments:
                for comment in submission.comments.list():
                    if hasattr(comment, 'body'):
                        output_item["comments"].append(comment.body)
            
            outputs.append(output_item)

        return outputs