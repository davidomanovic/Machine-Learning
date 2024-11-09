"""
We download the XLM-R model from the predefined torchtext 
models by following the instructions in torchtext.models. 
We also set the DEVICE to execute on-accelerator tests. 
(Enable GPU execution for your environment as appropriate.)
"""

import torch, torchtext, torch.nn
from torchtext.models import RobertaClassificationHead
from torchtext.functional import to_tensor
xlmr_large = torchtext.models.XLMR_LARGE_ENCODER
classifier_head = torchtext.models.RobertaClassificationHead(num_classes = 2, input_dim = 1024)
model = xlmr_large.get_model(head=classifier_head)
transform = xlmr_large.transform()

small_input = [
    "Hello World",
    "Goodbye World"
]

big_input = [
    "Hello world",
    "How are you!",
    """`Well, Prince, so Genoa and Lucca are now just family estates of the
Buonapartes. But I warn you, if you don't tell me that this means war,
if you still try to defend the infamies and horrors perpetrated by
that Antichrist- I really believe he is Antichrist- I will have
nothing more to do with you and you are no longer my friend, no longer
my 'faithful slave,' as you call yourself! But how do you do? I see
I have frightened you- sit down and tell me all the news.`

It was in July, 1805, and the speaker was the well-known Anna
Pavlovna Scherer, maid of honor and favorite of the Empress Marya
Fedorovna. With these words she greeted Prince Vasili Kuragin, a man
of high rank and importance, who was the first to arrive at her
reception. Anna Pavlovna had had a cough for some days. She was, as
she said, suffering from la grippe; grippe being then a new word in
St. Petersburg, used only by the elite."""
]

input_batch = big_input #Preprocess the inputs and test model
model_input = to_tensor(transform(input_batch), padding_value = 1)
output = model(model_input)
print(output.shape)

N = 10 #number of iterations

# Execution - Run and benchmark inference of CPU with and without BT fastpath
print("Slow path:")
with torch.autograd.profiler.profile(use_cuda=False) as prof:
    for i in range(N):
        output = model(model_input)

print(prof)
model.eval()

print("fast path:")
with torch.autograd.profiler.profile(use_cuda=False) as prof:
    with torch.no_grad():
        for i in range(N):
            output = model(model_input)

print(prof)