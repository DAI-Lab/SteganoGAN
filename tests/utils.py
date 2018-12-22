# -*- coding: utf-8 -*-

import torch


def assert_called_with_tensors(mock, calls):
    assert mock.call_count == len(calls), "Different number of calls"

    for mock_call, real_call in zip(mock.call_args_list, calls):
        assert len(mock_call) == len(real_call) - 1, "Different number of arguments"

        for mock_arg, arg in zip(mock_call[0], real_call[1]):
            if isinstance(mock_arg, torch.Tensor):
                assert (mock_arg == arg).all()
            else:
                assert mock_arg == arg
