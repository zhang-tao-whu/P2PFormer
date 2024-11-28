from ..plugins.msdeformattn_pixel_decoder import MSDeformAttnPixelDecoder

from ..builder import NECKS
@NECKS.register_module()
class MSDeformAttnFPN(MSDeformAttnPixelDecoder):
    pass