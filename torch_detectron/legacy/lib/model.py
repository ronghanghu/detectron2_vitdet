import torchvision.models as models

# Used to get the backbone network for the detection model
def backbone():
    # for now, just use VGG-13 as described in paper
    vgg = models.vgg13(pretrained=True)

    # we only care about the features, not the classifier
    return vgg.features

# Used to get the region proposal network for the detection model
def rpn():
    print('not implemented')

if __name__ == '__main__':
    print(backbone())
    
