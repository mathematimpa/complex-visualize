#!/usr/bin/env python3

from mayavi import mlab
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.colors as mcolors

# hue maps that actually look good

cmap = mcolors.LinearSegmentedColormap

D = np.pi/256
d1 = -np.pi/2
d2 = d1 - 2*np.pi/3
d3 = d1 + 2*np.pi/3
sinebow_table = {
'red'   : [(i/512, 0.5*np.cos(D*i+d1)+0.5 , 0.5*np.cos(D*i+d1)+0.5) for i in range(513)],
'green' : [(i/512, 0.5*np.cos(D*i+d2)+0.5 , 0.5*np.cos(D*i+d2)+0.5) for i in range(513)],
'blue'  : [(i/512, 0.5*np.cos(D*i+d3)+0.5 , 0.5*np.cos(D*i+d3)+0.5) for i in range(513)]
}
sinebow = cmap("sinebow", sinebow_table, N = 512)
plt.register_cmap(cmap=sinebow)

t = np.linspace(0, 2*np.pi, 255)
R = 0.5*np.cos(t)+0.5
G = 0.5*np.cos(t - 2*np.pi/3)+0.5
B = 0.5*np.cos(t + 2*np.pi/3)+0.5
A = np.ones_like(t)
rgba = 255*np.dstack((R,G,B,A)).reshape((-1, 4))

R = np.zeros((255,))
G = np.zeros((255,)) + 127
B = np.zeros((255,)) + 255
A = np.zeros((255,)) + 255
blue = np.dstack((R,G,B,A)).reshape((-1, 4))

# Misc

homotopy = lambda f, g : lambda t : lambda z : f(z)*(1-t) + g(z)*t

def blend_(c, v):
    if v < 0.5:
        return 2*c*v
    return 1 + 2*(1-c)*(v-1)

blend = np.vectorize(blend_)

def build_rect_grid(xrange, yrange):
    xmin, xmax, xsteps = xrange
    ymin, ymax, ysteps = yrange
    x, y = np.mgrid[xmin:xmax:1j*xsteps, ymin:ymax:1j*ysteps]
    return x + 1j*y

def build_polar_grid(rrange, trange):
    rmin, rmax, rsteps = rrange
    tmin, tmax, tsteps = trange
    r, th = np.mgrid[rmin:rmax:1j*rsteps, tmin:tmax:1j*tsteps]
    return r*np.exp(1j*th)

quina_ = lambda x : 1/max(np.abs(np.cos(x)), np.abs(np.sin(x)))
quina = np.vectorize(quina_)

build_polar = lambda rmax, rsteps, tsteps : build_polar_grid((0, rmax, rsteps), (0, 2*np.pi, tsteps))

def build_rect_polar_intergrid(R, rsteps, tsteps):
    r, th = np.mgrid[0:R:1j*rsteps, 0:2*np.pi:1j*tsteps]
    polar = r*np.exp(1j*th)
    rect  = polar*quina(th)
    return rect, polar

# Grid Mapping

def grid_plot(func, xrange, yrange):
    xmin, xmax, xsteps = xrange
    ymin, ymax, ysteps = yrange
    xvals = np.linspace(xmin, xmax, xsteps)
    yvals = np.linspace(ymin, ymax, ysteps)
    x     = np.linspace(xmin, xmax, 500)
    y     = np.linspace(ymin, ymax, 500)
    for x_ in xvals:
        f = func(x_ + 1j*y)
        plt.plot(f.real, f.imag, "-k")
    for y_ in yvals:
        f = func(x + 1j*y_)
        plt.plot(f.real, f.imag, "-k")

def grid_plot_colored(func, xrange, yrange):
    xmin, xmax, xsteps = xrange
    ymin, ymax, ysteps = yrange
    xvals = np.linspace(xmin, xmax, xsteps)
    yvals = np.linspace(ymin, ymax, ysteps)
    x     = np.linspace(xmin, xmax, 500)
    y     = np.linspace(ymin, ymax, 500)
    for x_ in xvals:
        f = func(x_ + 1j*y)
        plt.plot(f.real, f.imag, "-b")
    for y_ in yvals:
        f = func(x + 1j*y_)
        plt.plot(f.real, f.imag, "-r")

def grid_plot_colored_dots(func, xrange, yrange):
    xmin, xmax, xsteps = xrange
    ymin, ymax, ysteps = yrange
    xvals = np.linspace(xmin, xmax, xsteps)
    yvals = np.linspace(ymin, ymax, ysteps)
    x     = np.linspace(xmin, xmax, 500)
    y     = np.linspace(ymin, ymax, 500)
    for x_ in xvals:
        f = func(x_ + 1j*y)
        plt.plot(f.real, f.imag, "-b", zorder = 1)
    for y_ in yvals:
        f = func(x + 1j*y_)
        plt.plot(f.real, f.imag, "-r", zorder = 2)
    fvals = np.array([func(x + 1j*y) for x in xvals for y in yvals])
    plt.scatter(fvals.real, fvals.imag, c = "k", zorder = 3)

# Domain Coloring

class DColor:
    def __init__(self, samples_per_dot=50, xmin=-16, xmax=16, ymin=-9, ymax=9, step=5, gamma = 1, gap = 1):
        mpl.rcParams['toolbar'] = 'None'
        self._samples_per_dot = samples_per_dot
        #axes
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax
        self.step  = step
        self.gap   = gap
        self.makeDomain()
        self.gamma = gamma
    
    def makeDomain(self):
        """Create the domains for Real (x) and Imaginary (y) values respectively"""
        x = np.linspace(self._xmin, self._xmax, int(self._samples_per_dot)*(self._xmax-self._xmin))
        y = np.linspace(self._ymin, self._ymax, int(self._samples_per_dot)*(self._ymax-self._ymin))
        self.xx, self.yy=np.meshgrid(x,y)
    
    def makeColorModel(self, zz):
        A = np.angle(zz)
        R = 0.5*np.cos(A)+0.5
        G = 0.5*np.cos(A - 2*np.pi/3)+0.5
        B = 0.5*np.cos(A + 2*np.pi/3)+0.5
        r = np.log2(self.gap + np.abs(zz))
        V = ((1. + np.abs(np.cos(2. * np.pi * r / self.step))) / 2.)**.5
        return np.dstack((R*V,G*V,B*V))
    
    def makeColorModelBanded(self, zz):
        A = np.angle(zz)
        R = 0.5*np.cos(A)+0.5
        G = 0.5*np.cos(A - 2*np.pi/3)+0.5
        B = 0.5*np.cos(A + 2*np.pi/3)+0.5
        r = np.log2(self.gap + np.abs(zz))
        V = ((1. + np.abs(np.cos(2. * np.pi * r / self.step))) / 2.)**.5
        V*= ((1. + np.abs(np.cos(3 * A))) / 2.)**.5
        return np.dstack((R*V,G*V,B*V))
    
    def makeColorModelChroma(self, zz):
        A = np.angle(zz)
        R = 0.5*np.cos(A)+0.5
        G = 0.5*np.cos(A - 2*np.pi/3)+0.5
        B = 0.5*np.cos(A + 2*np.pi/3)+0.5
        r = np.log2(self.gap + np.abs(zz))
        V = (r/(r+1))**self.gamma
        return np.dstack((blend(R,V), blend(G,V), blend(B,V)))
    
    def makeColorModelChromaContour(self, zz):
        A = np.angle(zz)
        R = 0.5*np.cos(A)+0.5
        G = 0.5*np.cos(A - 2*np.pi/3)+0.5
        B = 0.5*np.cos(A + 2*np.pi/3)+0.5
        r = np.log2(self.gap + np.abs(zz))
        V = (r/(r+1))**self.gamma
        R, G, B = blend(R,V), blend(G,V), blend(B,V)
        V = ((1. + np.abs(np.cos(2. * np.pi * r / self.step))) / 2.)**.5
        return np.dstack((R*V, G*V, B*V))
    
    def normalize(self, arr):
        """Used for normalizing data in array based on min/max values"""
        arrMin = np.min(arr)
        arrMax = np.max(arr)
        arr = arr - arrMin
        return arr / (arrMax - arrMin)
    
    def plot(self, f, xdim=16, ydim=9, plt_dpi=120, name = "teste.png", model = "contour"):
        """Plot a complex-valued function
            Arguments:
            f -- a (preferably) lambda-function defining a complex-valued function
            Keyword Arguments:
            xdim -- x dimensions
            ydim -- y dimensions
            plt_dpi -- density of pixels per inch
        """
        zz=f(self.z(self.xx,self.yy))
        if model == "contour":
            rgb = self.makeColorModel(zz)
        elif model == "banded":
            rgb = self.makeColorModelBanded(zz)
        elif model == "chroma":
            rgb = self.makeColorModelChroma(zz)
        else:
            rgb = self.makeColorModelChromaContour(zz)
        fig = plt.figure(figsize=(xdim, ydim), dpi=plt_dpi)
        plt.axis('off')
        ax = fig.add_subplot(111)
        ax.imshow(rgb)
        ax.invert_yaxis() # make CCW orientation positive
        ax.set_aspect("equal")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(name)
        plt.close(fig)
    
    def z(self, x, y):
        """return complex number x+iy
            If inputs are arrays, then it returns an array with corresponding x_j+iy_j values
        """
        return x+1j*y

# 3D stuff

def plot_mesh_abs(func, grid, name = "teste.png", extent = None, fig = {"bgcolor" : (1, 1, 1), "fgcolor" : (0, 0, 0), "size" : (400, 300)}, view = {"focalpoint" : (0, 0, 0)}):
    x, y = grid.real, grid.imag
    f    = func(grid)
    z    = np.abs(f)
    s    = np.angle(f)
    mesh = {}
    if extent is not None:
        mesh["extent"] = extent
    mlab.figure(1, **fig)
    mlab.clf()
    surf = mlab.mesh(x, y, z, scalars = s, **mesh)
    surf.module_manager.scalar_lut_manager.lut.table = rgba
    mlab.view(**view)
    mlab.savefig(name)

def plot_mesh_proj(func, grid, name = "teste.png", theta = 0, extent = None, fig = {"bgcolor" : (1, 1, 1), "fgcolor" : (0, 0, 0), "size" : (400, 300)}, view = {"focalpoint" : (0, 0, 0)}):
    grid *= np.exp(1j*theta)
    x, y = grid.real, grid.imag
    f    = func(grid)
    z    = f.real
    mesh = {}
    if extent is not None:
        mesh["extent"] = extent
    mlab.figure(1, **fig)
    mlab.clf()
    surf = mlab.mesh(x, y, z, **mesh)
    surf.module_manager.scalar_lut_manager.lut.table = blue
    mlab.view(**view)
    mlab.savefig(name)

def plot_mesh_proj_hue(func, grid, name = "teste.png", theta = 0, extent = None, fig = {"bgcolor" : (1, 1, 1), "fgcolor" : (0, 0, 0), "size" : (400, 300)}, view = {"focalpoint" : (0, 0, 0)}):
    grid *= np.exp(1j*theta)
    x, y = grid.real, grid.imag
    f    = func(grid)
    z    = f.real
    s    = np.angle(f)
    mesh = {}
    if extent is not None:
        mesh["extent"] = extent
    mlab.figure(1, **fig)
    mlab.clf()
    surf = mlab.mesh(x, y, z, scalars = s, **mesh)
    surf.module_manager.scalar_lut_manager.lut.table = rgba
    mlab.view(**view)
    mlab.savefig(name)

def plot_mesh_general(zfunc, hfunc, grid, name = "teste.png", theta = 0, extent = None, fig = {"bgcolor" : (1, 1, 1), "fgcolor" : (0, 0, 0), "size" : (400, 300)}, view = {"focalpoint" : (0, 0, 0)}):
    grid *= np.exp(1j*theta)
    x, y = grid.real, grid.imag
    z    = zfunc(grid)
    s    = hfunc(grid)
    mesh = {}
    if extent is not None:
        mesh["extent"] = extent
    mlab.figure(1, **fig)
    mlab.clf()
    surf = mlab.mesh(x, y, z, scalars = s, **mesh)
    surf.module_manager.scalar_lut_manager.lut.table = rgba
    mlab.view(**view)
    mlab.savefig(name)

def change_view_and_save(view, name):
    mlab.view(**view)
    mlab.savefig(name)

interpolate_view = lambda v1, v2 : lambda t : {
"azimuth" : v2["azimuth"]*t + v1["azimuth"]*(1-t), 
"elevation" : v2["elevation"]*t + v1["elevation"]*(1-t),
"distance" : v2["distance"]*t + v1["distance"]*(1-t),
"focalpoint" : (v2["focalpoint"][0]*t + v1["focalpoint"][0]*(1-t), v2["focalpoint"][1]*t + v1["focalpoint"][1]*(1-t), v2["focalpoint"][2]*t + v1["focalpoint"][2]*(1-t))}

rotate_view = lambda v, th : {"azimuth" : v["azimuth"] + th, "elevation" : v["elevation"], "distance" : v["distance"], "focalpoint" : v["focalpoint"]}

