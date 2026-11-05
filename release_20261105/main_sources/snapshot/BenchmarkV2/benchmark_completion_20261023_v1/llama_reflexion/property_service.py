"""Native Llama property/feedback path; unknown predictor errors stop the unit."""
from common import *

if __name__ == '__main__':
    verify()
    sys.path.insert(0, str(ROOT.parent / 'recovery_20261027_v1'))
    from property_repair import install
    service = load_module('frozen_reflexion_properties', ENGINE / 'property_service.py')
    install(service)
    service.main()
