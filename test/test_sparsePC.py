
from neuralanalysis.visualization_ca import plotting_pcs
import numpy as np
import numpy.testing

def test_sparse():
    clu = np.array([0, 1])
    fet_ind = np.array(([0, 12, 14, 2, 10, 3, 0], [0, 12, 14, 2, 10, 3, 0]))
    fet = np.array(
        (
            [
                [6.03075, 7.51328, 2.29162, 0.637997, -1.4542, 0.443041, 4.31619],
                [
                    1.66104,
                    -0.343693,
                    -1.22469,
                    -0.0662854,
                    0.0165626,
                    -0.905885,
                    3.07569,
                ],
                [3.15145, 2.5234, -2.54112, 0.113551, -1.33154, -2.31246, -0.372592],
            ],
            [
                [11.5723, -1.30386, -0.88831, -4.61045, -2.05544, -0.384842, -3.10064],
                [-2.61024, 2.82361, 2.13182, -2.68165, -0.309417, -0.294074, 0.2321],
                [3.62874, 3.0533, -2.56696, 3.14891, -0.356746, 0.551253, 0.701876],
            ],
        )
    )


    sparse_pc = plotting_pcs.sparsePCs(fet, fet_ind, clu, 4, 15)
    assert np.shape(sparse_pc)==(2,45), "Incorrect shape of sparse_pc features"
    mean_pc = np.mean(sparse_pc, axis=1)
    numpy.testing.assert_allclose(mean_pc, np.array([0.471602,0.148473]),rtol=1e-05)