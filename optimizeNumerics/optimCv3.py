#!/Users/hkarimi/anaconda/lib/python3.5

##################################### LOAD MODULES #####################################
    # for performance timing
import time # use as: tic = time.time(); elapsedTime = time.time() - tic
    # numerical modules
import numpy as np
from numpy import sin,cos,sqrt
from scipy import integrate, optimize, interpolate
    # for plotting (with latex)
import matplotlib.pyplot as plt # main plotting module
plt.rc('text',usetex=True) # for latex
plt.rc('font',family='serif') # for latex
    # for sending data to MATLAB
import scipy.io as sio

# treating division by 0
np.seterr(divide='ignore')

################ optimize for drag coefficients ################

############### global variables ###############
##### universal constants
g = 9.81 # gravity constant, m/s^2

##### plate details
W = 0.02 # width (m)
L = 0.05914 # length (m)
stripColor = "red"
hdict = {"purple": 0.038/1000, "green": 0.08/1000, "red": 0.051/1000, "tan": 0.1/1000} # thickness (m)
Bdict = {"purple":2.5e-5, "green":1.7e-4, "red":7.5e-5, "tan":4e-4} # bending stiffness in Nm
h = hdict[stripColor]
B = Bdict[stripColor]
rhoPlate = 1.4697*(10**3) # density of plate, kg/m^3

##### towing velocity
U = 0.9/100 # m/s

##### fluid details
rhoFluid = 868 # density of fluid, kg/m^3
kinVisc = 2.54 # kinematic viscosity in Stokes = 10^-4 m^2 /s
mu = kinVisc * (10**-4) * rhoFluid # dynamic viscosity, kg/(m s)

##### non-dimensional parameters
Re = rhoFluid * U * L / mu
beta = (rhoPlate-rhoFluid)*g*h*L**3 / B # buyoancy constant
K = L**2 * U*mu / B # normal drag constant
rep = 1 # power of inverse Re in local drag force
Dt = rhoFluid * U**2 * L**3 / (B*(Re**(rep))) # tangential drag constant
#Dt = Dn

# load experimental data
Ucms = U*100
Lmm = L*1000
Wcm = W*100
loadfile = "U{:.2}L{:.4}.mat".format(Ucms,Lmm)
expData = sio.loadmat(loadfile)
xExp = expData['y']
yExp = -expData['x']
spl = interpolate.UnivariateSpline(xExp,yExp,k=4)

def odeSoln( coefficients ): # cn = normal drag coefficient, ct = tangential drag coefficient
    cn = coefficients[0]
    ct = coefficients[1]
    ct = abs(ct)
    ##### solve the ode #####
        # set up grid
    sBar = np.linspace(0,1,num=250)
    dsBar = sBar[1]-sBar[0]
        # set up LHS
    matSize = np.array([np.size(sBar),np.size(sBar)])
    LHS = np.zeros(matSize, dtype=float)
         # boundary conditions
    LHS[0,0] = 1 # BC at sBar = 0: theta = 0
    LHS[-1,-3:] = np.array([1, -4, 3]) / (2*dsBar) # at sBar = 1: dtheta_dsBar = 0
        # second-order central finite difference
    for n in range(1,np.size(sBar)-1):
        LHS[n,n-1] = 1/dsBar**2
        LHS[n,n] = -2/dsBar**2
        LHS[n,n+1] = 1/dsBar**2
        # set up RHS
    RHS = np.zeros(np.size(sBar))
        # bondary conditions
    RHS[0] = 0 # at sBar = 0: theta = 0
    RHS[-1] = 0 # at sBar = 1: dtheta_dsBar = 0
        # set up iteration
    thetaOld = np.linspace(0,1,num=np.size(sBar)) # initial guess
    itErr = 1 # initiate error
    itCount = 0 # iteration count
    weightOld = 7 # relaxation factor, in favor of thetaOld
    while (itErr > dsBar**2):
        itCount = itCount + 1 # update iteration count
        # set up RHS with thetaOld
            # normal vector in inertial frame (x,y)
        nx = cos(thetaOld)
        ny = sin(thetaOld)
            # external forcing / length
        Px = cn*K*cos(thetaOld)**2
        Py = cn*K*sin(thetaOld)*cos(thetaOld) - beta*np.ones(np.size(sBar))
            # account for gap between clamp and oil surface
        Px[sBar < 5.5/1000 / L] = 0
        Py[sBar < 5.5/1000 / L] = 0
            # integrated external force / length -> internal force
        Fx0 = integrate.trapz(Px,sBar)
        Fy0 = integrate.trapz(Py,sBar)
        Fx = Fx0 - integrate.cumtrapz(Px,sBar,initial=0) + ct*K*sin(thetaOld[-1])**2
        Fy = Fy0 - integrate.cumtrapz(Py,sBar,initial=0) - ct*K*sin(thetaOld[-1])*cos(thetaOld[-1])
            # fill up RHS
        for ns in range(1,np.size(sBar)-1): # internal ode, edges use BC's
            RHS[ns] = -( nx[ns]*Fx[ns] + ny[ns]*Fy[ns] )
        # solve for thetaNew
        thetaNew = np.linalg.solve(LHS,RHS)
        # check error and convergence criteria
        err = np.divide( (thetaNew-thetaOld) , thetaOld ) * dsBar/sBar[-1]
        err[np.isnan(err)] = 0 # remove NaN's from err, if divided by 0 for example
        err[np.isinf(err)] = 0
        itErr = sqrt(np.sum(np.dot(err,err)))
        # update for next iteration
        if itCount > 500:
            plt.plot(sBar,thetaNew,'-',sBar,thetaOld,'--')
            thetaOld = ( thetaNew + weightOld*thetaOld ) / (1+weightOld)
            print("extra plots since itCount > 500.")
            break
        thetaOld = ( thetaNew + weightOld*thetaOld ) / (1+weightOld)
    #elapsedTime = time.time() - tic
    #print('Time to complete', itCount,'iterations is {:4.3f}'.format(elapsedTime),'seconds.')
    print("itCount = {:g}, (cn,ct) = ({:g},{:g})".format(itCount,cn,ct))
    theta = thetaNew
    ############# kinematics to get xBar and yBar from theta ################
    xBar = integrate.cumtrapz(sin(theta),sBar,initial=0)
    yBar = integrate.cumtrapz(-cos(theta),sBar,initial=0)
    return xBar,yBar,sBar

def errNumExp(coefficients):
    xBar,yBar,sBar = odeSoln(coefficients)
    # match the tip of numerics to the splined experimental data
    delx = xBar[-1]*L*100 - xExp[-1]
    dely = yBar[-1]*L*100 - spl(xExp)[-1]
    xNum = xBar*L*100 - delx
    yNum = yBar*L*100 - dely
    yExpComp = np.interp(xNum, xExp[:,0], spl(xExp)[:,0] )
    yErr = (yNum -yExpComp)[sBar > 5.5/1000] # ignore the first 10 mm, since 5.5 mm is a gap
    compErr = np.linalg.norm(yErr) / len(yErr)
#    print("comptErr = {:g}".format(compErr))
    return compErr
    
############# optimize for ct and cn ################
# initial guesses
cn = 28.0
ct = 20.0
# initiate timer
tic = time.time()

#lowBounds = [20,15] # lower bounds of [cn,ct]
#hiBounds = [30,25] # upper bounds of [cn,ct[]
    # solve by least_squares
#optimResult = optimize.least_squares( errNumExp, np.array([cn,ct]) , bounds = ( lowBounds,hiBounds ) )
    # solve by Powell method (better for global search)
optimResult = optimize.minimize(errNumExp,np.array([cn,ct]),method='Powell',bounds=( (0,100),(0,100) ))
print(optimResult)
cn,ct = optimResult.x

    # solve by brute force
#rranges = (slice(25, 31, 0.25), slice(16, 22, 0.25))
#resbrute = optimize.brute(errNumExp, rranges, full_output=True, finish=optimize.least_squares)
#cn,ct = resbrute[0]  # global minimum

c = [cn,ct]
elapsedTime = time.time() - tic
print("Elapsed time of optimization = {:g} seconds".format(elapsedTime))

#%% ######################## plotting the solution ##########################
xBar,yBar,sBar = odeSoln( [cn,ct] )
# match the tip of numerics to the splined experimental data
delx = xBar[-1]*L*100 - xExp[-1]
dely = yBar[-1]*L*100 - spl(xExp)[-1]
xNumPlot = xBar*L*100 - delx
yNumPlot = yBar*L*100 - dely
fig,ax = plt.subplots(1,1,figsize=(4,4))
#ax.plot(xExp,yExp,xExp,spl(xExp),xNumPlot,yNumPlot)
ax.plot(xExp,spl(xExp),xNumPlot-0.0/100,yNumPlot)
ax.plot( [-0.5, 2.5], (yNumPlot[0]-5.5/10) * np.array([1, 1]) )
ax.set_xlabel(r'x [cm]')
ax.set_ylabel(r'y [cm]')
ax.set_xlim([-0.5,L*100])
ax.set_ylim([-L*100,1])
ax.legend(('smooth exp','numerics'))
#ax.legend(('raw','smooth exp','numerics'))
ax.grid(True)

fig.savefig("plot.pdf", format='pdf')

print("(cn,ct) = ({:g},{:g})".format(cn,ct))
########## save data to compare to experiments ##########
#yExp = L*xBar.copy()
#xExp = L*yBar.copy()
#sio.savemat( 'np_xy.mat', { 'x':xExp ,'y':yExp } )
