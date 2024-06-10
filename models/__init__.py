from .model1 import SequenceNUCNet1
from .model2 import SequenceNUCNet2
from .model3 import SequenceNUCNet3
from .model4 import UNetConvLSTM


model_dict =  {
    'model1' : SequenceNUCNet1,
    'model2' : SequenceNUCNet2,
    'model3' : SequenceNUCNet3,
    'model4' : UNetConvLSTM
}