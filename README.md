## Spending Programmed Bidding

This repository contains the bidding strategy source code for Spending Programmed Bidding: Privacy-friendly Bid Optimization with ROI Constraint in Online Advertising. It uses a modified [Auction Gym](https://github.com/amzn/auction-gym) as simulation environment.

## Framework Modification

To illustrate SPB strategy, this work makes the following modifications to the Auction Gym framework:

- Introduce conversion(click) feedback delay and aggregating simulate restrictions from privacy regulations.
- Add perturbation on conversion(click) rate estimation to simulate unstable estimation in privacy protection enviroment.
- Add a budget restriction for each iteration

## Environment Settings

- Each bidder bids for a single campaign, thus each bidder other than the environment bidder have only one item and its value is fixed over several budget cycle
- Each iteration is treated as a budget cycle for the campaign
- The performance of bidding result is evaluated according to the standard descriped in the Spending Programmed Bidding paper.

## Run Simulation

To simulate with SPB strategy, run:

```
python src/main.py config/SPB.json
```
This will simulation one campaign using SPB strategy against an environment for several budget cycle.

