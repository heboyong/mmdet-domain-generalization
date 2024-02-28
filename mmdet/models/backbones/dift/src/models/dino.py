# from extractor_dino import ViTExtractor
import torch
from torch import nn
import os, sys
from functools import partial

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from thirdparty.dinov2.dinov2.eval.linear import get_args_parser, ModelWithIntermediateLayers, setup_linear_classifiers, setup_and_build_model


class DinoWithLinear(nn.Module):
    def __init__(self, autocast_dtype):

        n_last_blocks_list = [1, 4]
        n_last_blocks = max(n_last_blocks_list)
        autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
        feature_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx)
        # sample_output = feature_model(train_dataset[0][0].unsqueeze(0).cuda())
        sample_output = torch.rand(2, 384, 25, 25)
        batch_size = 8
        training_num_classes = 19
        learning_rates = 0.0005

        linear_classifiers, optim_param_groups = setup_linear_classifiers(
            sample_output,
            n_last_blocks_list,
            learning_rates,
            batch_size,
            training_num_classes,
        )

        self.classifier = setup_linear_classifiers()


if __name__ == '__main__':

    # model_type = 'dinov2_vits14'
    # extractor = ViTExtractor(model_type=model_type, stride=14)
    # print(extractor.p)
    # img_size = 512
    # stride = extractor.stride[0]
    # patch_size = extractor.model.patch_embed.patch_size[0]
    # print(int(patch_size / stride * (img_size // patch_size - 1) + 1))

    # x = torch.rand(1, 3, 512, 512)
    # y = extractor.extract_descriptors(x, facet='token')
    
    # print(y.shape)

    from transformers import AutoImageProcessor, AutoModel
    from PIL import Image
    import requests

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    processor = AutoImageProcessor.from_pretrained('/home/xmuairmud/jyx/dinov2-base')
    model = AutoModel.from_pretrained('/home/xmuairmud/jyx/dinov2-base')

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs, output_hidden_states=True)
    last_hidden_states = outputs.last_hidden_state

    print(inputs.pixel_values.shape)
    for hidden_state in outputs.hidden_states:
        print(hidden_state.shape)

    