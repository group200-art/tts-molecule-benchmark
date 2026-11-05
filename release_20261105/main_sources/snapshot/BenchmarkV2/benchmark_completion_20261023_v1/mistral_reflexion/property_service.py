"""Persistent original property producers, redirected into the continuation."""
import importlib
import sys
from runtime import ROOT, bind, verify_freeze

if __name__ == '__main__':
    verify_freeze()
    bind()
    sys.path.insert(0, str(ROOT.parent / 'recovery_20261027_v1'))
    from property_repair import install
    service = importlib.import_module('property_service')
    install(service)
    service.main()
