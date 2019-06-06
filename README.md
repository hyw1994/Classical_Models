# Classical_Models
This is the project to build classical deep learning models for research purpose.

# ResNet: make deeper network easier to train.
ResNet implicitly adds the input to the output of the group of layers. This action promises that the output will work at least as an identical layer.
## Problems Review
Deep networks are suffering from multiple obstacles. Although gradient vanishing/exploding problems can be solved by proper initialization and normalization techniques, the accuracy won't increase along with the depth of the network gradually. Such **degradation** problem is not caused by overfitting because the training error is also higher than shallower network. 

Instead of setting the layers to learn the underlying features directly, we explicitly let the layers to learn the residual part of the features. **It means that, we want the layers to learn the differences between the last layer.**

## Network Architecture
### Basic Building Block
<div align= center>
<img src="imgs/ResNet_Building_Blocks.png" width=80%/>
<b><i>

ResNet Building Block
</i></b>
</div>