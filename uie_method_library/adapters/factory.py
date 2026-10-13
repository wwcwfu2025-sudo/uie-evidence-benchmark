from __future__ import annotations

from uie_method_library.core import AdapterError, MethodAdapter, MethodSpec

from .classical import ClassicalAdapter
from .autoenhancer import AutoEnhancerAdapter
from .cluienet import CLUIENetAdapter
from .dicam import DICAMAdapter
from .deep_wavenet import DeepWaveNetAdapter
from .funiegan import FunieGanAdapter
from .histoformer import HistoformerAdapter
from .luieo import LuieoAdapter
from .p2cnet import P2CNetAdapter
from .puienet import PUIENetAdapter
from .sguie import SGUIEAdapter
from .shallow_uwnet import ShallowUWNetAdapter
from .spectroformer import SpectroformerAdapter
from .tacl import TACLAdapter
from .tctlnet import TCTLNetAdapter
from .udnet import UdnetAdapter
from .unavailable import UnavailableAdapter
from .ushape_transformer import UShapeTransformerAdapter
from .uwtoolbox import UwEnhancementToolboxAdapter
from .uvenet import UveNetAdapter


def build_adapter(spec: MethodSpec, device: str = "auto") -> MethodAdapter:
    if spec.adapter.startswith("classical:"):
        return ClassicalAdapter(spec, device)
    if spec.adapter == "autoenhancer":
        return AutoEnhancerAdapter(spec, device)
    if spec.adapter == "cluienet":
        return CLUIENetAdapter(spec, device)
    if spec.adapter == "dicam":
        return DICAMAdapter(spec, device)
    if spec.adapter == "deep_wavenet":
        return DeepWaveNetAdapter(spec, device)
    if spec.adapter == "funiegan":
        return FunieGanAdapter(spec, device)
    if spec.adapter == "histoformer":
        return HistoformerAdapter(spec, device)
    if spec.adapter == "luieo":
        return LuieoAdapter(spec, device)
    if spec.adapter == "p2cnet":
        return P2CNetAdapter(spec, device)
    if spec.adapter == "puienet":
        return PUIENetAdapter(spec, device)
    if spec.adapter == "sguie":
        return SGUIEAdapter(spec, device)
    if spec.adapter == "shallow_uwnet":
        return ShallowUWNetAdapter(spec, device)
    if spec.adapter == "spectroformer":
        return SpectroformerAdapter(spec, device)
    if spec.adapter == "tacl":
        return TACLAdapter(spec, device)
    if spec.adapter == "tctlnet":
        return TCTLNetAdapter(spec, device)
    if spec.adapter == "udnet":
        return UdnetAdapter(spec, device)
    if spec.adapter == "uvenet":
        return UveNetAdapter(spec, device)
    if spec.adapter == "uwtoolbox":
        return UwEnhancementToolboxAdapter(spec, device)
    if spec.adapter == "ushape_transformer":
        return UShapeTransformerAdapter(spec, device)
    if spec.adapter in ("pending", "external_pending", "uve_net_pending"):
        return UnavailableAdapter(spec, device)
    raise AdapterError("Unknown adapter type: %s" % spec.adapter)
