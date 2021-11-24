from essmc2.utils.typing import check_seq_of_seq


def test_check_seq_of_seq():
    assert not check_seq_of_seq([])
    assert check_seq_of_seq([[1]])
    assert check_seq_of_seq([(1, )])
    assert not check_seq_of_seq([(1, ), 1])
    assert not check_seq_of_seq((1, ))
    assert check_seq_of_seq(((1, ), (2, )))
