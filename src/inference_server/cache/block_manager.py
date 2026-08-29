
from collections import deque
import dataclasses from dataclass


@dataclass
class Block:
    id : int

    owner : int | None = None

class BlockManager:

    def __init__(self, num_blocks: int):

        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_blocks = deque(range(num_blocks))

    def allocate(self,request_id):

        if not self.free_blocks:
            return None

        block_id = self.free_blocks.popleft()

        self.blocks[block_id].owner  = request_id

    def free(self,block_id):

        self.blocks[block].owner = None

        self.free_blocks.append(block_id)