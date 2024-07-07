#!/usr/bin/env python
"""
Summary
-------
Provide dictionary directories listing solvers, problems, and models.
"""

# import solvers
from .solvers.astrodf import ASTRODF
from .solvers.randomsearch import RandomSearch
from .solvers.neldmd import NelderMead
from .solvers.strong import STRONG
from .solvers.spsa import SPSA
from .solvers.adam import ADAM
from .solvers.aloe import ALOE

# import models and problems
from .models.example import ExampleModel, ExampleProblem
from .models.cntnv import CntNV, CntNVMaxProfit
from .models.mm1queue import MM1Queue, MM1MinMeanSojournTime
from .models.facilitysizing import (
    FacilitySize,
    FacilitySizingTotalCost,
    FacilitySizingMaxService,
)
from .models.rmitd import RMITD, RMITDMaxRevenue
from .models.sscont import SSCont, SSContMinCost
from .models.ironore import IronOre, IronOreMaxRev, IronOreMaxRevCnt
from .models.dynamnews import DynamNews, DynamNewsMaxProfit
from .models.dualsourcing import DualSourcing, DualSourcingMinCost
from .models.contam import (
    Contamination,
    ContaminationTotalCostDisc,
    ContaminationTotalCostCont,
)
from .models.chessmm import ChessMatchmaking, ChessAvgDifference
from .models.san import SAN, SANLongestPath
from .models.hotel import Hotel, HotelRevenue
from .models.tableallocation import TableAllocation, TableAllocationMaxRev
from .models.paramesti import ParameterEstimation, ParamEstiMaxLogLik
from .models.fixedsan import FixedSAN, FixedSANLongestPath
from .models.network import Network, NetworkMinTotalCost
from .models.amusementpark import AmusementPark, AmusementParkMinDepart

# Import base
from .base import Model, Problem, Solver

# directory dictionaries
solver_directory: dict[str, "Solver"] = {
    "ASTRODF": ASTRODF,
    "RNDSRCH": RandomSearch,
    "NELDMD": NelderMead,
    "STRONG": STRONG,
    "SPSA": SPSA,
    "ADAM": ADAM,
    "ALOE": ALOE,
}

solver_unabbreviated_directory: dict[str, "Solver"] = {
    "ASTRO-DF (SBCN)": ASTRODF,
    "Random Search (SSMN)": RandomSearch,
    "Nelder-Mead (SBCN)": NelderMead,
    "STRONG (SBCN)": STRONG,
    "SPSA (SBCN)": SPSA,
    "ADAM (SBCN)": ADAM,
    "ALOE (SBCN)": ALOE,
}

solver_unabbreviated_directory: dict[str, str] = {
    "ASTRO-DF (SBCN)": "ASTRODF",
    "Random Search (SSMN)": "RNDSRCH",
    "Nelder-Mead (SBCN)": "NELDMD",
    "STRONG (SBCN)": "STRONG",
    "SPSA (SBCN)": "SPSA",
    "ADAM (SBCN)": "ADAM",
    "ALOE (SBCN)": "ALOE",
}

solver_introduction_directory: dict[str, str] = {
    "ASTRO-DF (SBCN)": "ASTRO-DF (SBCN): The ASTRO-DF solver progressively builds local models (quadratic with diagonal Hessian) using interpolation on a set of points on the coordinate bases of the best (incumbent) solution. Solving the local models within a trust region (closed ball around the incumbent solution) at each iteration suggests a candidate solution for the next iteration. If the candidate solution is worse than the best interpolation point, it is replaced with the latter (a.k.a. direct search). The solver then decides whether to accept the candidate solution and expand the trust-region or reject it and shrink the trust-region based on a success ratio test. The sample size at each visited point is determined adaptively and based on closeness to optimality.",
    "Random Search (SSMN)": "Random Search (SSMN): The Random Search solver randomly sample solutions from the feasible region. Can handle stochastic constraints.",
    "Nelder-Mead (SBCN)": "Nelder-Mead (SBCN): An algorithm that maintains a simplex of points that moves around the feasible region according to certain geometric operations: reflection, expansion, contraction, and shrinking.",
    "STRONG (SBCN)": "STRONG (SBCN): A trust-region-based algorithm that fits first- or second-order models through function evaluations taken within a neighborhood of the incumbent solution.",
    "SPSA (SBCN)": "SPSA (SBCN): Simultaneous perturbation stochastic approximation (SPSA) is an algorithm for optimizing systems with multiple unknown parameters.",
    "ADAM (SBCN)": "ADAM (SBCN): An algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments.",
    "ALOE (SBCN)": "ALOE (SBCN): The solver is a stochastic line search algorithm  with the gradient estimate recomputed in each iteration, whether or not a step is accepted. The algorithm includes the relaxation of the Armijo condition by an additive constant.",
}

problem_directory: dict[str, "Problem"] = {
    "EXAMPLE-1": ExampleProblem,
    "CNTNEWS-1": CntNVMaxProfit,
    "MM1-1": MM1MinMeanSojournTime,
    "FACSIZE-1": FacilitySizingTotalCost,
    "FACSIZE-2": FacilitySizingMaxService,
    "RMITD-1": RMITDMaxRevenue,
    "SSCONT-1": SSContMinCost,
    "IRONORE-1": IronOreMaxRev,
    "IRONORECONT-1": IronOreMaxRevCnt,
    "DYNAMNEWS-1": DynamNewsMaxProfit,
    "DUALSOURCING-1": DualSourcingMinCost,
    "CONTAM-1": ContaminationTotalCostDisc,
    "CONTAM-2": ContaminationTotalCostCont,
    "CHESS-1": ChessAvgDifference,
    "SAN-1": SANLongestPath,
    "HOTEL-1": HotelRevenue,
    "TABLEALLOCATION-1": TableAllocationMaxRev,
    "PARAMESTI-1": ParamEstiMaxLogLik,
    "FIXEDSAN-1": FixedSANLongestPath,
    "NETWORK-1": NetworkMinTotalCost,
    "AMUSEMENTPARK-1": AmusementParkMinDepart,
}

problem_unabbreviated_directory: dict[str, "Problem"] = {
    "Min Deterministic Function + Noise (SUCG)": ExampleProblem,
    "Max Profit for Continuous Newsvendor (SBCG)": CntNVMaxProfit,
    "Min Mean Sojourn Time for MM1 Queue (SBCG)": MM1MinMeanSojournTime,
    "Min Total Cost for Facility Sizing (SSCG)": FacilitySizingTotalCost,
    "Max Service for Facility Sizing (SDCN)": FacilitySizingMaxService,
    "Max Revenue for Revenue Management Temporal Demand (SDDN)": RMITDMaxRevenue,
    "Min Total Cost for (s, S) Inventory (SBCN)": SSContMinCost,
    "Max Revenue for Iron Ore (SBDN)": IronOreMaxRev,
    "Max Revenue for Continuous Iron Ore (SBCN)": IronOreMaxRevCnt,
    "Max Profit for Dynamic Newsvendor (SBDN)": DynamNewsMaxProfit,
    "Min Cost for Dual Sourcing (SBDN)": DualSourcingMinCost,
    "Min Total Cost for Discrete Contamination (SSDN)": ContaminationTotalCostDisc,
    "Min Total Cost for Continuous Contamination (SSCN)": ContaminationTotalCostCont,
    "Min Avg Difference for Chess Matchmaking (SSCN)": ChessAvgDifference,
    "Min Mean Longest Path for Stochastic Activity Network (SBCG)": SANLongestPath,
    "Max Revenue for Hotel Booking (SBDN)": HotelRevenue,
    "Max Revenue for Restaurant Table Allocation (SDDN)": TableAllocationMaxRev,
    "Max Log Likelihood for Gamma Parameter Estimation (SBCN)": ParamEstiMaxLogLik,
    "Min Mean Longest Path for Fixed Stochastic Activity Network (SBCG)": FixedSANLongestPath,
    "Min Total Cost for Communication Networks System (SDCN)": NetworkMinTotalCost,
    "Min Total Departed Visitors for Amusement Park (SDDN)": AmusementParkMinDepart,
}
model_directory: dict[str, "Model"] = {
    "EXAMPLE": ExampleModel,
    "CNTNEWS": CntNV,
    "MM1": MM1Queue,
    "FACSIZE": FacilitySize,
    "RMITD": RMITD,
    "SSCONT": SSCont,
    "IRONORE": IronOre,
    "DYNAMNEWS": DynamNews,
    "DUALSOURCING": DualSourcing,
    "CONTAM": Contamination,
    "CHESS": ChessMatchmaking,
    "SAN": SAN,
    "HOTEL": Hotel,
    "TABLEALLOCATION": TableAllocation,
    "PARAMESTI": ParameterEstimation,
    "FIXEDSAN": FixedSAN,
    "NETWORK": Network,
    "AMUSEMENTPARK": AmusementPark,
}
model_unabbreviated_directory: dict[str, "Model"] = {
    "Deterministic Function + Noise": ExampleModel,
    "Continuous Newsvendor": CntNV,
    "MM1 Queue": MM1Queue,
    "Facility Sizing": FacilitySize,
    "Revenue Management Temporal Demand": RMITD,
    "(s, S) Inventory": SSCont,
    "Iron Ore": IronOre,
    "Dynamic Newsvendor": DynamNews,
    "Dual Sourcing": DualSourcing,
    "Contamination": Contamination,
    "Chess Matchmaking": ChessMatchmaking,
    "Stochastic Activity Network": SAN,
    "Hotel Booking": Hotel,
    "Restaurant Table Allocation": TableAllocation,
    "Gamma Parameter Estimation": ParameterEstimation,
    "Fixed Stochastic Activity Network": FixedSAN,
    "Communication Networks System": Network,
    "Amusement Park (SDDN)": AmusementPark,
}
model_problem_unabbreviated_directory: dict[str, str] = {
    "Min Deterministic Function + Noise (SUCG)": "EXAMPLE",
    "Max Profit for Continuous Newsvendor (SBCG)": "CNTNEWS",
    "Min Mean Sojourn Time for MM1 Queue (SBCG)": "MM1",
    "Min Total Cost for Facility Sizing (SSCG)": "FACSIZE",
    "Max Service for Facility Sizing (SDCN)": "FACSIZE",
    "Max Revenue for Revenue Management Temporal Demand (SDDN)": "RMITD",
    "Min Total Cost for (s, S) Inventory (SBCN)": "SSCONT",
    "Max Revenue for Iron Ore (SBDN)": "IRONORE",
    "Max Revenue for Continuous Iron Ore (SBCN)": "IRONORE",
    "Max Profit for Dynamic Newsvendor (SBDN)": "DYNAMNEWS",
    "Min Cost for Dual Sourcing (SBDN)": "DUALSOURCING",
    "Min Total Cost for Discrete Contamination (SSDN)": "CONTAM",
    "Min Total Cost for Continuous Contamination (SSCN)": "CONTAM",
    "Min Avg Difference for Chess Matchmaking (SSCN)": "CHESS",
    "Min Mean Longest Path for Stochastic Activity Network (SBCG)": "SAN",
    "Max Revenue for Hotel Booking (SBDN)": "HOTEL",
    "Max Revenue for Restaurant Table Allocation (SDDN)": "TABLEALLOCATION",
    "Max Log Likelihood for Gamma Parameter Estimation (SBCN)": "PARAMESTI",
    "Min Mean Longest Path for Fixed Stochastic Activity Network (SBCG)": "FIXEDSAN",
    "Min Total Cost for Communication Networks System (SDCN)": "NETWORK",
    "Min Total Departed Visitors for Amusement Park (SDDN)": "AMUSEMENTPARK",
}
model_problem_class_directory: dict[str, "Model"] = {
    "Min Deterministic Function + Noise (SUCG)": ExampleModel,
    "Max Profit for Continuous Newsvendor (SBCG)": CntNV,
    "Min Mean Sojourn Time for MM1 Queue (SBCG)": MM1Queue,
    "Min Total Cost for Facility Sizing (SSCG)": FacilitySize,
    "Max Service for Facility Sizing (SDCN)": FacilitySize,
    "Max Revenue for Revenue Management Temporal Demand (SDDN)": RMITD,
    "Min Total Cost for (s, S) Inventory (SBCN)": SSCont,
    "Max Revenue for Iron Ore (SBDN)": IronOre,
    "Max Revenue for Continuous Iron Ore (SBCN)": IronOre,
    "Max Profit for Dynamic Newsvendor (SBDN)": DynamNews,
    "Min Cost for Dual Sourcing (SBDN)": DualSourcing,
    "Min Total Cost for Discrete Contamination (SSDN)": Contamination,
    "Min Total Cost for Continuous Contamination (SSCN)": Contamination,
    "Min Avg Difference for Chess Matchmaking (SSCN)": ChessMatchmaking,
    "Min Mean Longest Path for Stochastic Activity Network (SBCG)": SAN,
    "Max Revenue for Hotel Booking (SBDN)": Hotel,
    "Max Revenue for Restaurant Table Allocation (SDDN)": TableAllocation,
    "Max Log Likelihood for Gamma Parameter Estimation (SBCN)": ParameterEstimation,
    "Min Mean Longest Path for Fixed Stochastic Activity Network (SBCG)": FixedSAN,
    "Min Total Cost for Communication Networks System (SDCN)": Network,
    "Min Total Departed Visitors for Amusement Park (SDDN)": AmusementPark,
}

model_introduction_directory: dict[str, str] = {
    "EXAMPLE": "A model that is a deterministic function evaluated with noise.",
    "CNTNEWS": "A model that simulates a day's worth of sales for a newsvendor with a Burr Type XII demand distribution. Returns the profit, after accounting for order costs and salvage.",
    "MM1": "    A model that simulates an M/M/1 queue with an Exponential(lambda) interarrival time distribution and an Exponential(x) service time distribution. Returns - the average sojourn time- the average waiting time- the fraction of customers who wait for customers after a warmup period.",
    "FACSIZE": "A model that simulates a facilitysize problem with a multi-variate normal distribution. Returns the probability of violating demand in each scenario.",
    "RMITD": "A model that simulates a multi-stage revenue management system with inter-temporal dependence. Returns the total revenue.",
    "SSCONT": "A model that simulates multiple periods' worth of sales for a (s,S) inventory problem with continuous inventory, exponentially distributed demand, and poisson distributed lead time. Returns the various types of average costs per period, order rate, stockout rate, fraction of demand met with inventory on hand, average amount backordered given a stockout occured, and average amount ordered given an order occured.",
    "IRONORE": "A model that simulates multiple periods of production and sales for an inventory problem with stochastic price determined by a mean-reverting random walk. Returns total profit, fraction of days producing iron, and mean stock.",
    "DYNAMNEWS": "A model that simulates a day's worth of sales for a newsvendor with dynamic consumer substitution. Returns the profit and the number of products that stock out.",
    "DUALSOURCING": "A model that simulates multiple periods of ordering and sales for a single-staged, dual sourcing inventory problem with stochastic demand. Returns average holding cost, average penalty cost, and average ordering cost per period.",
    "CONTAM": "A model that simulates a contamination problem with a beta distribution. Returns the probability of violating contamination upper limit in each level of supply chain.",
    "CHESS": "A model that simulates a matchmaking problem with a Elo (truncated normal) distribution of players and Poisson arrivals. Returns the average difference between matched players.",
    "SAN": "A model that simulates a stochastic activity network problem with tasks that have exponentially distributed durations, and the selected means come with a cost.",
    "HOTEL": "A model that simulates business of a hotel with Poisson arrival rate.",
    "TABLEALLOCATION": "A model that simulates a table capacity allocation problem at a restaurant with a homogenous Poisson arrvial process and exponential service times. Returns expected maximum revenue.",
    "PARAMESTI": "A model that simulates MLE estimation for the parameters of a two-dimensional gamma distribution.",
    "FIXEDSAN": "A model that simulates a stochastic activity network problem with tasks that have exponentially distributed durations, and the selected means come with a cost.",
    "NETWORK": "Simulate messages being processed in a queueing network.",
    "AMUSEMENTPARK": "    A model that simulates a single day of operation for an amusement park queuing problem based on a poisson distributed tourist arrival rate, a next attraction transition matrix, and attraction durations based on an Erlang distribution. Returns the total number and percent of tourists to leave the park due to full queues.",
}
