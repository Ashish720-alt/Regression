

smallPercents = [90, 95, 100, 105, 110 ] #10% is considered large change
largeIncrPercents = [150, 200, 500, 1000]
largeDecrPercents = [50, 1, 0.1, 0.01]


# Useful for hyperparameters whose values are unbounded and can take large values like learning rate, ridge coefficient.
def relativelyChanged (baseConfig, percentChange, isIncrease):
    change =  (percentChange / 100 ) * baseConfig
    if (isIncrease):
        return baseConfig + change
    else:
        return max(0, baseConfig - change)
    
smallAbsoluteChange = [ 1, 10, 20]
largeAbsoluteChange = [50, 100, 500, 1000]


# Useful for those hyperparameters whose ranges are bounded, like degree etc
def absoluteChanged(baseConfig, absChange, isIncrease):
    if (isIncrease):
        return baseConfig + absChange
    else:
        return baseConfig - absChange