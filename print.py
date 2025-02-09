

def printLinearRegressionModel( coefficients , intercept, precision=4):
    print("\tLearned function:")
    print(f"\ty = {intercept:.{precision}f} ", end="")
    for i, coef in enumerate(coefficients):
        print(f"+ ({coef:.{precision}f}) * X{i+1} ", end="")
    print('\n')
    return


def printError( trainError, testError, errorType, precision=4, isval = 1):
    errString = "(test)" if not isval else "(val)"
    print('\t' + errorType + "(train)" + f": {trainError:.{precision}f}" + ' , ' + errorType + errString + f": {testError:.{precision}f}")