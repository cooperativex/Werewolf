from typing import List
from uuid import uuid1

from .message_item import Message


class MessagePool:
    """
    A pool to manage the messages in the chatArena environment.

    The pool is essentially a list of messages, and it allows a unified treatment of the visibility of the messages.
    It supports two configurations for step definition: multiple players can act in the same turn (like in rock-paper-scissors).
    Agents can only see the messages that 1) were sent before the current turn, and 2) are visible to the current role.
    """

    def __init__(self):
        """
        Initialize the MessagePool with a unique conversation ID.
        """
        self.conversation_id = str(uuid1())
        self._messages: List[Message] = []  # TODO: for the sake of thread safety, use a queue instead
        self._last_message_idx = 0

    def reset(self):
        """
        Clear the message pool.
        """
        self._messages = []

    def append_message(self, message: Message):
        """
        Append a message to the pool.

        Parameters:
            message (Message): The message to be added to the pool.
        """
        self._messages.append(message)

    def print(self):
        """
        Print all the messages in the pool.
        """
        for message in self._messages:
            print(f"[{message.agent_name}->{message.visible_to}]: {message.content}")

    @property
    def last_turn(self):
        """
        Get the turn of the last message in the pool.

        Returns:
            int: The turn of the last message.
        """
        if len(self._messages) == 0:
            return 0
        else:
            return self._messages[-1].turn

    @property
    def last_message(self):
        """
        Get the last message in the pool.

        Returns:
            Message: The last message.
        """
        if len(self._messages) == 0:
            return None
        else:
            return self._messages[-1]

    def get_all_messages(self) -> List[Message]:
        """
        Get all the messages in the pool.

        Returns:
            List[Message]: A list of all messages.
        """
        return self._messages

    def get_visible_messages(self, agent_name, turn: int) -> List[Message]:
        """
        Get all the messages that are visible to a given agent before a specified turn.

        Parameters:
            agent_name (str): The name of the agent.
            turn (int): The specified turn.

        Returns:
            List[Message]: A list of visible messages.
        """

        # Get the messages before the current turn
        prev_messages = [message for message in self._messages if message.turn < turn]

        visible_messages = []
        for message in prev_messages:
            if message.visible_to == "all" or agent_name in message.visible_to or agent_name == "Moderator":
                visible_messages.append(message)
        return visible_messages
