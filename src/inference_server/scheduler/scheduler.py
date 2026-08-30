
from inference_server.generation.generator import Generator
from inference_server.model.hf_model import HFModel
from inference_server.cache.block_manager import BlockManager
from inference_server.scheduler.state import RequestState

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
                continue

            elif request.state == RequestState.PREFILL:

                token = generator.prefill(request)

                if token:
                    tokens[request.id] = token

            elif request.state == RequestState.DECODING:

                if generator.need_new_block(request):

                    block = self.block_manager.allocate(request.id)

                    if block is None:

                        request.state = RequestState.WAITING_FOR_BLOCK
                    
                        continue

                    request.block_table.append(block)

                
                token = generator.decode(request)
                if token:
                    tokens[request.id] = token

            elif request.state == RequestState.WAITING_FOR_BLOCK:

                block = self.block_manager.allocate(request.id)

                if block is not None:
                    request.block_table.append(block)
                    request.state = RequestState.DECODING

                    token = generator.decode(request)

                    if token:
                        tokens[request.id] = token

        for request in self.finished:
            for block in request.block_table:
                self.block_manager.free(block)

            request.block_table.clear()
            self.remove(request)

        self.finished.clear()


        return tokens

        
                