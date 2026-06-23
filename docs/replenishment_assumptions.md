# Replenishment Decision-Support Assumptions

## Purpose

The replenishment module converts the 16-day demand forecast into a
deterministic ordering recommendation.

It is a decision-support simulation and does not represent actual
Corporación Favorita inventory operations.

## User Inputs

The calculation requires:

- current inventory;
- confirmed inbound inventory;
- supplier lead time;
- safety-stock days;
- inventory review period;
- case-pack size;
- minimum order quantity.

These values are assumptions supplied by the user. They are not included
in the original Kaggle dataset.

## Formulas

Inventory position:

`current inventory + inbound inventory`

Lead-time demand:

`sum of forecast demand during the lead-time period`

Safety stock:

`average forecast daily demand × safety-stock days`

Reorder point:

`lead-time demand + safety stock`

Target inventory:

`forecast demand during lead time and review period + safety stock`

Raw order quantity:

`maximum of zero and target inventory minus inventory position`

The final suggested quantity is adjusted to satisfy the minimum order
quantity and case-pack size.

## Risk Bands

- Critical: inventory is insufficient to cover expected lead-time demand.
- High: inventory is at or below the reorder point.
- Moderate: inventory exceeds the reorder point but is below the target.
- Low: inventory is at or above the target.

These bands are deterministic rules. They are not probabilities of
stockout.

## Limitations

The calculation does not currently model:

- supplier reliability;
- forecast uncertainty intervals;
- spoilage;
- product shelf life;
- ordering cost;
- holding cost;
- lost-sales cost;
- minimum display stock;
- warehouse capacity.

The recommendation should therefore support, rather than replace, human
inventory planning.
