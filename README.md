# Robotic Assisted PCNL
This repository contains scripts and data used for the work *Ultrasonographic-Guided Robotic-Assisted Percutaneous Nephrolithotomy*.
The work aims to develop a system for providing help to a surgeon who needs to perform renal access during a Percutaneous Nephrolitotomy (PCNL) surgery.
Starting from pre-operative CT images, organs and andatomical structures are reconstructed to automatically plan a safe trajectory for the needle.
During intra-operative phase, a collaborative robotic arm equipped with an Ultrasound probe and a needle guide is exploited for positioning in the pose that allows surgeon to insert needle along the needle guide following the pre-planned trajectory.

## Data
[data](https://github.com/stefanoBazzani/robotic_assisted_pcnl/tree/main/data) contains both CT volume and US images of the phantom used during tests and experiments.

## Code
In [code](https://github.com/stefanoBazzani/robotic_assisted_pcnl/tree/main/code) folder you can find scripts used for the pre-operative trajectory planning.
