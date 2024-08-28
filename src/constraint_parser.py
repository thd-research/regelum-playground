import numpy as np
from regelum.constraint_parser import ConstraintParser, linear_constraint
import regelum as rg
from typing import List, Optional, Generator
from functools import partial


# class QcarConstraintParser(ConstraintParser):
#     def __init__(
#         self,
#         centers: Optional[np.ndarray] = None,
#         coefs: Optional[np.ndarray] = None,
#         radii: Optional[np.ndarray] = None,
#         weights: Optional[np.ndarray] = None,
#         biases: Optional[np.ndarray] = None,
#     ) -> None:
#         """Instantiate CylindricalHalfPlaneConstraintParser.

#         Args:
#             centers (Optional[np.ndarray], optional): centers of
#                 circles, defaults to None
#             coefs (Optional[np.ndarray], optional): circle coeficients,
#                 defaults to None
#             radii (Optional[np.ndarray], optional): radii of circles,
#                 defaults to None
#             weights (Optional[np.ndarray], optional): lines'
#                 coefficients, defaults to None
#             biases (Optional[np.ndarray], optional): lines' biases,
#                 defaults to None
#         """
#         self.radii = np.array(radii) if radii is not None else radii
#         self.centers = np.array(centers) if centers is not None else centers
#         self.coefs = np.array(coefs) if coefs is not None else coefs
#         self.weights = np.array(weights) if weights is not None else weights
#         self.biases = np.array(biases) if biases is not None else biases

#     def _parse_constraints(self, simulation_metadata=None):
#         return {
#             "radii": self.radii,
#             "centers": self.centers,
#             "coefs": self.coefs,
#             "weights": self.weights,
#             "biases": self.biases,
#         }

#     def constraint_parameters(self):
#         return [
#             self.ConstraintParameter(name, data.shape, data)
#             for name, data in [
#                 ["radii", self.radii],
#                 ["centers", self.centers],
#                 ["coefs", self.coefs],
#                 ["weights", self.weights],
#                 ["biases", self.biases],
#             ]
#             if data is not None
#         ]

#     def constraint_function(
#         self,
#         weights=None,
#         biases=None,
#         radii=None,
#         centers=None,
#         coefs=None,
#         predicted_states=None,
#     ):
#         weights = weights if weights is not None else self.weights
#         biases = biases if biases is not None else self.biases
#         radii = radii if radii is not None else self.radii
#         centers = centers if centers is not None else self.centers
#         coefs = coefs if coefs is not None else self.coefs

#         linear_constraint_values = None
#         circle_constraint_values = None
#         if weights is not None and biases is not None:
#             linear_constraint_values = linear_constraint(
#                 weights=weights, bias=biases, predicted_states=predicted_states
#             )
#         if radii is not None and centers is not None and coefs is not None:
#             circle_constraint_values = circle_constraint(
#                 coefs=coefs,
#                 radius=radii,
#                 center=centers,
#                 predicted_states=predicted_states,
#             )
#         constraint_values = rg.vstack(
#             [
#                 val
#                 for val in [linear_constraint_values, circle_constraint_values]
#                 if val is not None
#             ]
#         )
#         return rg.max(constraint_values)
