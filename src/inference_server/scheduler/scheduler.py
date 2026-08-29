
from inference_server.generation.generator import Generator
from inference_server.model.HFModel import HFModel
from inference_mode.cache.block_manager import BlockManager

class Scheduler:

    def __init__(self,block_manager):

        self.block_manager = block_manager    
        self.finished = []
        self.requests = []


    def add(self, request):
        self.requests.append(request)


    def remove(self, request):
        self.requests.remove(request)

    def has_requests(self):

        return len(self.requests) > 0

    def step(self, generator : Generator):

        if not self.requests:
            return {}
        tokens = {}

        for request in self.requests:

            if request.finished:

                self.finished.append(request)

            elif request.state == PREFILL:

                token = generator.prefill(request)

                if token:
                    tokens[request.id] = token

            elif request.state == DECODING:

                if generator.need_new_block(request):

                    block = self.block_manager.allocate(request.id)

                    if block is None:

                        request.state = WAITING_FOR_BLOCK
                    
                        continue

                    request.block_table.append(block)

                
                token = generator.decode(request)
                if token:
                    tokens[request.id] = token

            elif request.state == WAITING_FOR_BLOCK:

                block = self.block_manager.allocate(request.id)

                if block is not None:
                    request.block_table.append(block)
                    request.state = DECODING

        for request in self.finished:

            for block in request.block_table:
                self.block_manager.free(block)

            request.block_table.clear()


        return tokens

        
                