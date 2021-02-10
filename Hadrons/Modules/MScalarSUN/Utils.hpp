/*
 * Utils.hpp, part of Hadrons (https://github.com/aportelli/Hadrons)
 *
 * Copyright (C) 2015 - 2020
 *
 * Author: Antonin Portelli <antonin.portelli@me.com>
 *
 * Hadrons is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * Hadrons is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Hadrons.  If not, see <http://www.gnu.org/licenses/>.
 *
 * See the full license in the file "LICENSE" in the top level distribution 
 * directory.
 */

/*  END LEGAL */
#ifndef Hadrons_MScalarSUN_Utils_hpp_
#define Hadrons_MScalarSUN_Utils_hpp_

#include <Hadrons/Global.hpp>
#include <Hadrons/Module.hpp>

BEGIN_HADRONS_NAMESPACE

BEGIN_MODULE_NAMESPACE(MScalarSUN)

GRID_SERIALIZABLE_ENUM(DiffType, undef, forward, 1, backward, 2, central, 3);

template <typename Field>
inline void dmu(Field &out, const Field &in, const unsigned int mu, const DiffType type)
{
    auto & env = Environment::getInstance();

    if (mu >= env.getNd())
    {
        HADRONS_ERROR(Range, "Derivative direction out of range");
    }
    switch(type)
    {
        case DiffType::backward:
            out = in - Cshift(in, mu, -1);
            break;
        case DiffType::forward:
            out = Cshift(in, mu, 1) - in;
            break;
        case DiffType::central:
            out = 0.5*(Cshift(in, mu, 1) - Cshift(in, mu, -1));
            break;
        default:
            HADRONS_ERROR(Argument, "Derivative type invalid");
            break;
    }
}

template <typename Field>
inline void dmuAcc(Field &out, const Field &in, const unsigned int mu, const DiffType type)
{
    auto & env = Environment::getInstance();

    if (mu >= env.getNd())
    {
        HADRONS_ERROR(Range, "Derivative direction out of range");
    }
    switch(type)
    {
        case DiffType::backward:
            out += in - Cshift(in, mu, -1);
            break;
        case DiffType::forward:
            out += Cshift(in, mu, 1) - in;
            break;
        case DiffType::central:
            out += 0.5*(Cshift(in, mu, 1) - Cshift(in, mu, -1));
            break;
        default:
            HADRONS_ERROR(Argument, "Derivative type invalid");
            break;
    }
}

template <class SinkSite, class SourceSite>
std::vector<Complex> makeTwoPoint(const std::vector<SinkSite>   &sink,
                                  const std::vector<SourceSite> &source,
                                  const double factor = 1.)
{
    assert(sink.size() == source.size());
    
    unsigned int         nt = sink.size();
    std::vector<Complex> res(nt, 0.);
    
    for (unsigned int dt = 0; dt < nt; ++dt)
    {
        for (unsigned int t  = 0; t < nt; ++t)
        {
            res[dt] += trace(sink[(t+dt)%nt]*adj(source[t]));
        }
        res[dt] *= factor/static_cast<double>(nt);
    }
    
    return res;
}


inline double WindowBeta(double a, double b, double r)
{
    return exp(-1.0/(r-a)-1.0/(b-r));
}

//Integrate WindowGamma(r, a, c) from a to r
inline double trapezoidalIntegral(double a, double b, double r, int n, const std::function<double (double, double, double)> &f) {
    const double width = (r-a)/n;

    double trapezoidal_integral = 0;
    for(int step = 0; step < n; step++) {
        const double x1 = a + step*width;
        const double x2 = a + (step+1)*width;

        trapezoidal_integral += 0.5*(x2-x1)*(f(a, b, 1) + f(a, b, x2));
    }

    return trapezoidal_integral;
}

inline double windowFunction(double a, double b, double r, int n_step)
{
    if (r < a)
    {
        return 0.0;
    }
    else if (r > b)
    {
        return 1.0;
    } else
    {
        return trapezoidalIntegral(a, b, r, n_step, &WindowBeta) / trapezoidalIntegral(a, b, b, n_step, &WindowBeta);
    }
}

inline std::string varName(const std::string name, const std::string suf)
{
    return name + "_" + suf;
}

inline std::string varName(const std::string name, const unsigned int mu)
{
    return varName(name, std::to_string(mu));
}

inline std::string varName(const std::string name, const unsigned int mu, 
                           const unsigned int nu)
{
    return varName(name, std::to_string(mu) + "_" + std::to_string(nu));
}

END_MODULE_NAMESPACE

END_HADRONS_NAMESPACE

#endif // Hadrons_MScalarSUN_Utils_hpp_
