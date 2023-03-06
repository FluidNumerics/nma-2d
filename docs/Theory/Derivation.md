# Derivation of Normal Mode Analysis


## Problem definition & motivation
The normal mode analysis in `xnma` assumes that we are interested in calculating the spectra associated with a 2-D velocity field in an arbitrary domain. For logically rectangular domains with periodic boundary conditions, Fourier Transforms are well suited to calculate spectra. However, velocity fields in regional ocean/atmosphere simulations are not periodic. To handle the lack of periodicity, windowing is usually performed wherein the velocity fields are multiplied by a function that decays to zero smoothly near the domain boundaries. Additionally, 2-D planes of data formed by intersecting regional model domains with a plane at constant depth result in "islands" or "holes". To handle these irregularities, researchers usually interpolate through these discontinuities. These workarounds arose simply because we had not considered an alternative approach that is designed to handle irregular geometries and non-periodicity.


The goal of spectral analysis is to explain the distribution of energy across a range of length scales. This is done by projecting data onto a set of basis functions that are provably associated with specific length scales. In spectral analysis, we assume that the data we work with are observations of a square integrable functions. Because of this, the basis functions form a **basis** for all square integrable functions; this means that any square integrable function can be written as a linear combination of these basis functions. 

With Fourier Analysis, the basis functions are complex exponentials. In 2-D, the basis functions are obtained by multiplying by complex exponentials in $x$ and $y$ directions. When projecting non-periodic data onto periodic basis functions, the spectral slopes decay slowly (less than $\mathcal{O}(k^{1})$). This causes complications for researchers who wish to measure spectral slopes.

With Normal Mode Analysis, vector fields are written in terms of a Helmholz decomposition and the basis functions are generated from a Sturm-Liouville Boundary Value problem that incorporates regional geometry. Note that the Fourier Basis functions are a special case, where the Fourier modes are the solution to Laplace's equation on a rectangular domain with periodic boundary conditions. In the `xnma` methodology, the basis functions are solutions to Laplace's equation with homogeneous Dirichlet boundary conditions and the geometry is defined by the user-provided regional domain. The Helmholz decomposition splits the velocity field into divergent and rotational components and is used to ensure that the projection operation matches the boundary normal and tangential velocities. This approach allows us to ascribe energy to specific length scales and to rotational and divergent motions.


### Formal Definitions
To help formalize our discussion, let's define a few things

* The 2-D domain where we have velocity field data $A$ and its boundary is $\partial A$, 
* The space of square integrable functions on $A$ is denoted $L_2(A)$
* The set of functions that form a basis for $L_2(A)$  are written as $\{\xi_i(x,y)\}_{i=0}^{\infty}$

A square integrable function on $A$ is any function that satisfies

\begin{equation}
\int_A f(x,y)^2 \hspace{1mm} dA \hspace{5mm} \text{    is finite} 
\end{equation}

Functions that satisfy this condition are "in the space of square integrable functions"
\begin{equation}
f \in L_2(A)
\end{equation}

Any square integrable function on $A$ can be written as a linear combination of the basis functions for $L_2(A)$
\begin{equation}
 f(x,y) = \sum_{i=0}^{\infty} \hat{f}_i \xi_i(x,y)
\end{equation}

A square integrable vector function on $A$ is written as
\begin{equation}
\vec{u} = u \hat{x} + v \hat{y}; \hspace{5mm} u,v \in L_2(A)
\end{equation}

Any 2-D vector field can be written in terms of their rotational and divergent components; this is the **Helmholz Decomposition**. These are described by the gradients of two scalar functions

\begin{equation}
\vec{u} = ( -\Psi_y + \Phi_x ) \hat{x} + ( \Psi_x + \Phi_y ); \hspace{5mm} \Psi, \Phi \in L_2(A)
\end{equation}

Taking the divergence of $\vec{u}$ gives
\begin{equation}
\nabla \cdot \vec{u} = \nabla^2 \Phi
\end{equation}

Taking the curl of $\vec{u}$ gives
\begin{equation}
\nabla \times \vec{u} = \nabla^2 \Psi
\end{equation}


### Boundary conditions for the Helmholz potentials
The Helmholz decomposition results in inhomogeneous elliptic equations for the Helmholz potentials $\Psi$ and $\Phi$. Such a partial differential equation also needs boundary conditions to uniquely define a solution. For the boundary condition, we appeal to energetics.

For our purposes, the vector fields we are interested are velocity fields for which the kinetic energy is proportional to $||\vec{u}||^2$. The total area integrated energy is written as

\begin{equation}
E_k = \int_A u^2 + v^2 \hspace{1mm} dA = \int_A ||\nabla \Psi||^2 + ||\nabla \Phi||^2 - \nabla \Phi \times \nabla \Psi \hspace{1mm} dA 
\end{equation}

The last term on the right-hand-side, can be written in two different ways, by using the product rule and Stoke's theorem
\begin{equation}
\int_A \nabla \Phi \times \nabla \Psi \hspace{1mm} dA = \oint_{\partial A} \Phi \nabla \Psi \cdot \hat{t} \hspace{1mm} dS = - \oint_{\partial A} \Psi \nabla \Phi \cdot \hat{t} \hspace{1mm} dS
\end{equation}

If we let $\Psi|_{\partial A} = 0$ and $\Phi|_{\partial A} = 0$, then the total energy can be written as
\begin{equation}
E_k = \int_A u^2 + v^2 \hspace{1mm} dA = \int_A ||\nabla \Psi||^2 + ||\nabla \Phi||^2 \hspace{1mm} dA 
\end{equation}

This is a desirable result in that the total energy can be cleanly separated in terms of the rotational energy $||\nabla \Psi||^2$ and divergent energy $||\nabla \Phi||^2$. Because of this, the boundary value problems for the Helmholz potentials are

\begin{align}
\nabla^2 \Phi &= \nabla \cdot \vec{u} \\
\Phi|_{\partial A}  &= 0
\end{align}

\begin{align}
\nabla^2 \Psi &= \nabla \times \vec{u} \\
\Psi|_{\partial A} &= 0
\end{align}


### Series solutions to the Helmholz problems
Series solutions to boundary value problems are expressed as a linear combination of basis functions. $\Phi$ and $\Psi$ both solve elliptic boundary value problems with homogeneous Dirichlet boundary conditions; their main difference is in the "forcing" term on the right-hand-side. 

The appropriate basis functions to use in a series solution are generated from the Sturm-Liouville eigenvalue problem

\begin{align}
\nabla^2 \xi_i &= \lambda_i \xi_i \\
\xi_i|_{\partial A} &= 0
\end{align}


