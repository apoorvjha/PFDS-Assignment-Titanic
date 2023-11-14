import sys
import os
sys.path.append("../source")
import orchestrator
sys.path.pop()


def test_orchestrator():
    try:
        orchestrator.main()
        assert True
    except Exception as e:
        assert False