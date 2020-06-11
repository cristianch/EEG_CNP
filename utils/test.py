import unittest
from experiments_classification import get_tp_tn_fp_fn


class Test(unittest.TestCase):

    def test_get_tp_tn_fp_fn(self):
        true = [1, 1, 1, 0, 0, 0]
        pred = [1, 1, 1, 0, 1, 1]
        tp, tn, fp, fn = get_tp_tn_fp_fn(true, pred)
        self.assertEqual(tp, 3)
        self.assertEqual(tn, 1)
        self.assertEqual(fp, 2)
        self.assertEqual(fn, 0)

    if __name__ == '__main__':
        unittest.main()