from inference_server.scheduler.scheduler import Scheduler
from inference_server.cache.block_manager import BlockManager
from inference_server.scheduler.request import Request
from inference_server.generation.generator import Generator

from inference_server.model.hf_model import HFModel
from inference_server.sampler.sampler import Sampler
from inference_server.tokenizer.hf_tokenizer import HFTokenizer

block_manager = BlockManager(num_blocks=2)
scheduler = Scheduler(block_manager)


tokenizer = HFTokenizer()
model = HFModel()
sampler = Sampler()

generator = Generator(tokenizer,model,sampler)


request1 = Request(
    id=1,
    prompt="Hello",
    max_new_tokens=10,
)

request2 = Request(
    id=2,
    prompt="What is AI?",
    max_new_tokens=10,
)

request3 = Request(
    id=3,
    prompt="Tell me about Google",
    max_new_tokens=10,
)

tick = 0


scheduler.add(request1)
scheduler.add(request2)
scheduler.add(request3)

generator.initialize(request1)
generator.initialize(request2)
generator.initialize(request3)

MAX_TICKS = 20

while scheduler.has_requests() and tick < MAX_TICKS:

    tick += 1

    token = scheduler.step(generator)

    print("")
    print("")
    
    print(f"Tick : {tick}")
    print(f"token : {token}")

    for request in scheduler.requests:

        print(
            request.id,
            request.state,
            len(request.generated_ids),
            request.block_table,
        )    

    print("Free:", list(block_manager.free_blocks))
        