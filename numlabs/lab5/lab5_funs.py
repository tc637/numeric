import numpy as np
import yaml
from collections import namedtuple


def rkck_init():
    # %
    # % initialize the Cash-Karp coefficients
    # % defined in the tableau in lab 4,
    # % section "Embedded Runge Kutta"
    # %
    a = np.array([0.2, 0.3, 0.6, 1.0, 0.875])
  # c1 coefficients for the fifth order scheme
    c1 = np.array([37.0 / 378.0, 0.0, 250.0 / 621.0,
                   125.0 / 594.0, 0.0, 512.0 / 1771.0])
  # c2=c* coefficients for the fourth order schme
    c2 = np.array([2825.0 / 27648.0, 0.0, 18575.0 / 48384.0,
                   13525.0 / 55296.0, 277.0 / 14336.0, .25])
    b = np.empty([5, 5], 'float')
  # the following line is ci - ci* in lab4, \Delta_est equationl
  # this is used to calculate \Delta_est = estError for the embededd
  # Runge Kutta  \sum_^6 (c_i -c_i^*)
  #
    c2 = c1 - c2
  # this sets b values for same tableu 
    b[0, 0] = 0.2
    b[1, 0] = 3.0 / 40.0
    b[1, 1] = 9.0 / 40.0
    b[2, 0] = 0.3
    b[2, 1] = -0.9
    b[2, 2] = 1.2
    b[3, 0] = -11.0 / 54.0
    b[3, 1] = 2.5
    b[3, 2] = -70.0 / 27.0
    b[3, 3] = 35.0 / 27.0
    b[4, 0] = 1631.0 / 55296.0
    b[4, 1] = 175.0 / 512.0
    b[4, 2] = 575.0 / 13824.0
    b[4, 3] = 44275.0 / 110592.0
    b[4, 4] = 253.0 / 4096.0
    return (a, c1, c2, b)


class Integrator:

    def set_yinit(self):
        raise ValueError(
            'set_init needs to be overridden in the derived class')

    def __init__(self, coeffFileName):
        with open(coeffFileName, 'rb') as f:
            config = yaml.load(f)
        self.config = config
        # read in dt tstart tend
        timevars = namedtuple('timevars',config['timevars'].keys())
        self.timevars = timevars(**config['timevars'])
        # read in dtpassmin dtpassmax dtfailmin dtfailmax s rtol atol maxsteps maxfail
        adaptvars = namedtuple('adaptvars', config['adaptvars'].keys())
        self.adaptvars = adaptvars(**config['adaptvars'])
        self.rkckConsts = rkck_init()

    def __str__(self):
        out = 'integrator instance with attributes initvars, timevars,uservars, ' + \
            'adaptvars'
        return out

    def derivs5(self, y, t):
        raise ValueError('derivs5 needs to be overrideen in the derived class')
        return None

    def rkckODE5(self, yold, timeStep, deltaT):

        # initialize the Cash-Karp coefficients
        # defined in the tableau in lab 4,

        a, c1, c2, b = self.rkckConsts
        i = self.initvars
        # set up array to hold k values in lab4 
        derivArray = np.empty([6, self.nvars], 'float')
        ynext = np.zeros_like(yold)
        bsum = np.zeros_like(yold)
        estError = np.zeros_like(yold)
        # vector k1 in lab4 equation 3.9
        derivArray[0, :] = self.derivs5(yold, timeStep)[:]

        # calculate step
        # c1=c_i in lab 4 notation, but c2=c_i - c^*_i

        y = yold
        for i in np.arange(5):
            bsum = 0.
            for j in np.arange(i + 1):
                bsum = bsum + b[i, j] * derivArray[j, :]
            # vectors k2 through k6 in lab4 
#           pdb.set_trace()
            derivArray[i + 1, :] = self.derivs5(y + deltaT * bsum, timeStep + a[i] * deltaT)[:]
            # partial sum of error in lab4 \Delta_est
            #
            #  sum the error term
            #
            estError = estError + c2[i] * derivArray[i, :]
            # print "estError: ",estError
            #
            # 5th order estimate y_{n+1}
            #
            ynext = ynext + c1[i] * derivArray[i, :]
        # final fifth order anser
        y = y + deltaT * (ynext + c1[5] * derivArray[5, :])
        # final 4th order estimate estimate
        estError = deltaT * (estError + c2[5] * derivArray[5, :])
        # print "estError final: ",estError
        timeStep = timeStep + deltaT
#       pdb.set_trace()
        return (y, estError, timeStep)

    def timeloop5Err(self):
        """return errors as well as values
        """
        t = self.timevars
        a = self.adaptvars
        i = self.initvars
        nvars = self.nvars
        oldTime = t.tstart
        olddt = t.dt
        yold = self.yinit
        yerror = np.zeros_like(yold)
        num = 0
        badsteps = 0
        goodsteps = 0
        timeVals = []
        yvals = []
        errorList = []
        while(oldTime < t.tend):
            timeVals.append(oldTime)
            yvals.append(yold)
            errorList.append(yerror)
            if(num > a.maxsteps):
                raise Exception('num > maxsteps')
            # start out with goodstep false and
            # try different sizes for the next step
            # until one meets the error conditions
            # then move onto next step by setting
            # goodstep to true
            goodStep = False
            failSteps = 0
            while(not goodStep):
                # to exit this loop, need to
                # get the estimated error smaller than
                # the desired error set by the relative
                # tolerance
                if(failSteps > a.maxfail):
                    raise Exception('failSteps > a.maxfail')
                #
                # try a timestep, we may need to reverse this
                #
                ynew, yerror, timeStep = self.rkckODE5(yold, oldTime, olddt)
                # print("try a step: : ", ynew)
                #
                # lab 5 section 4.2.3
                # find the desired tolerance by multiplying the relative
                # tolerance (RTOL) times the value of y
                # compare this to the error estimate returnd from rkckODE5
                # atol takes care of the possibility that y~0 at some point
                #
                errtest = 0.
                for i in range(nvars):
                    errtest = errtest + \
                        (yerror[i] / (a.atol + a.rtol * np.abs(ynew[i])))**2.0
                errtest = np.sqrt(errtest / nvars)
                #
                # lab5 equation 4.13, S
                #
                dtchange = a.s * (1.0 / errtest)**0.2
                # print("dtchange, errtest, timeStep: ",
                #       dtchange, errtest, timeStep, ynew, yerror)
                if (errtest > 1.0):
                    # estimated error is too big so
                    # reduce the timestep and retry
                    # dtFailMax ~ 0.5, which guarantees that
                    # the new timestep is reduced by at least a
                    # factor of 2
                    # dtFailMin~0.1, which means that we don't trust
                    # the estimate to reduce the timestep by more
                    # than a factor of 10 in one loop
                    if(dtchange > a.dtfailmax):
                        olddt = a.dtfailmax * olddt
                    elif (dtchange < a.dtfailmin):
                        olddt = a.dtfailmin * olddt
                    else:
                        olddt = dtchange * olddt
                    if (timeStep + olddt == timeStep):
                        raise Exception('step smaller than machine precision')
                    failSteps = failSteps + 1
                    #
                    # undo the timestep since the error wasn't small enough
                    #
                    ynew = yold
                    timeStep = oldTime
                    # go back to top and see if this olddt produices
                    # a better yerrror
                else:
                    # errtest < 1, so we're happy
                    # try to enlarge the timestep by a factor of dtChange > 1
                    # but keep it smaller than dtpassmax
                    # try enlarging the timestep bigger for next time
                    # dtpassmin ~ 0.1 and dtpassmax ~ 5
                    if (abs((1.0 - dtchange)) > a.dtpassmin):
                        if(dtchange > a.dtpassmax):
                            dtnew = a.dtpassmax * olddt
                        else:
                            dtnew = dtchange * olddt
                    else:
                        # don't bother changing the step size if
                        # the change is less than dtpassmin
                        dtnew = olddt
                    goodStep = True
                    #
                    # overwrite the old timestep with the new one
                    #
                    oldTime = timeStep
                    yold = ynew
                    # go back up to top while(timeStep < t.tend)
                    goodsteps = goodsteps + 1
                #
                # this is number of times we decreased the step size without
                #  advancing
                #
                badsteps = badsteps + failSteps
            # special case if we're within one ortwo timesteps of the end
            # otherwise, set dt to the new timestep size
            if(timeStep + dtnew > t.tend):
                olddt = t.tend - timeStep
            elif(timeStep + 2.0 * dtnew > t.tend):
                olddt = (t.tend - timeStep) / 2.0
            else:
                olddt = dtnew
        timeVals = np.array(timeVals).squeeze()
        yvals = np.array(yvals).squeeze()
        errorVals = np.array(errorList).squeeze()
        self.timevals = timeVals
        self.yvals = yvals
        self.errorVals = errorVals
        return (timeVals, yvals, errorVals)

    def timeloop5fixed(self):
        """fixed time step with
           estimated errors
        """
        t = self.timevars
        yold = self.yinit
        yError = np.zeros_like(yold)
        yvals = [yold]
        errorList = [yError]
        timeSteps = np.arange(t.tstart, t.tend, t.dt)
        for theTime in timeSteps[:-1]:
            yold, yError, newTime = self.rkckODE5(yold, theTime, t.dt)
            yvals.append(yold)
            errorList.append(yError)
        yvals = np.array(yvals).squeeze()
        errorVals = np.array(errorList).squeeze()
        return (timeSteps, yvals, errorVals)


if __name__ == "__main__":
    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt

    theSolver = Integrator('example1/daisy.ini')

    timeVals, yvals, errList = theSolver.timeloop5Err()
    whiteDaisies = [frac[0] for frac in yvals]

    thefig = plt.figure(1)
    thefig.clf()
    theAx = thefig.add_subplot(111)
    points = theAx.plot(timeVals, whiteDaisies, 'b+')
    theLines = theAx.plot(timeVals, yvals)
    theLines[1].set_linestyle('--')
    theLines[1].set_color('k')
    theAx.set_title('lab 5 example 1')
    theAx.set_xlabel('time')
    theAx.set_ylabel('fractional coverage')
    theAx.legend(theLines, ('white daisies', 'black daisies'), loc='best')
    plt.show()
