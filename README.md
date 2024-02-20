## Spending Programmed Bidding

This repository contains the source code for the bidding strategy used in 'Spending Programmed Bidding: Privacy-friendly Bid Optimization with ROI Constraint in Online Advertising.' It employs a modified version of [Auction Gym](https://github.com/amzn/auction-gym) as the simulation environment.

## Framework Modification

To demonstrate the SPB strategy, we've made the following modifications to the Auction Gym framework:

- Introduced a delay in the feedback for conversions (clicks) and aggregation restrictions simulated from privacy regulations.
- Added perturbation to the estimation of conversion (click) rate to simulate unstable estimations in a privacy-protected environment.
- Implemented a budget restriction for each iteration.

## Environment Settings

- Each bidder bids for a single campaign; thus, each bidder, other than the environment bidder, has only one item, and its value is fixed over several budget cycles.
- Each iteration is treated as a budget cycle for the campaign.
- The performance of the bidding results is evaluated according to the standard described in the Spending Programmed Bidding paper.

## Run Simulation

To simulate with the SPB strategy, run:

```
python src/main.py config/SPB.json
```

This will simulate a campaign using the SPB strategy against an environment for several budget cycles.

