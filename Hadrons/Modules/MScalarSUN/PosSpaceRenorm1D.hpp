#ifndef Hadrons_MScalarSUN_PosSpaceRenorm1D_hpp_
#define Hadrons_MScalarSUN_PosSpaceRenorm1D_hpp_

#include <Hadrons/Global.hpp>
#include <Hadrons/Module.hpp>
#include <Hadrons/ModuleFactory.hpp>
#include <Hadrons/Modules/MScalarSUN/Utils.hpp>

BEGIN_HADRONS_NAMESPACE

/******************************************************************************
 *                         PosSpaceRenorm1D                                 *
 ******************************************************************************/
BEGIN_MODULE_NAMESPACE(MScalarSUN)

class PosSpaceRenorm1DPar: Serializable
{
public:
    typedef std::pair<std::string, std::string> OpPair;
    GRID_SERIALIZABLE_CLASS_MEMBERS(PosSpaceRenorm1DPar,
                                    double, zeroval,
                                    double, windowmin,
                                    double, windowmax,
                                    std::vector<OpPair>, op,
                                    std::vector<std::string>, mom,
                                    std::string, output);
};

class PosSpaceRenorm1DResult : Serializable
{
public:
    GRID_SERIALIZABLE_CLASS_MEMBERS(PosSpaceRenorm1DResult,
                                    std::string, sink,
                                    std::string, source,
                                    std::vector<int>, mom,
                                    std::vector<Complex>, data);
};

template <typename SImpl>
class TPosSpaceRenorm1D: public Module<PosSpaceRenorm1DPar>
{
public:
    typedef typename SImpl::Field Field;
    typedef typename SImpl::ComplexField ComplexField;
    typedef std::vector<Complex> SlicedOp;
public:
    // constructor
    TPosSpaceRenorm1D(const std::string name);
    // destructor
    virtual ~TPosSpaceRenorm1D(void) {};
    // dependency relation
    virtual std::vector<std::string> getInput(void);
    virtual std::vector<std::string> getOutput(void);
    // setup
    virtual void setup(void);
    // execution
    virtual void execute(void);
private:
    std::vector<std::vector<int>> mom_;
};

MODULE_REGISTER_TMP(PosSpaceRenorm1DSU2, TPosSpaceRenorm1D<ScalarNxNAdjImplR<2>>, MScalarSUN);
MODULE_REGISTER_TMP(PosSpaceRenorm1DSU3, TPosSpaceRenorm1D<ScalarNxNAdjImplR<3>>, MScalarSUN);
MODULE_REGISTER_TMP(PosSpaceRenorm1DSU4, TPosSpaceRenorm1D<ScalarNxNAdjImplR<4>>, MScalarSUN);
MODULE_REGISTER_TMP(PosSpaceRenorm1DSU5, TPosSpaceRenorm1D<ScalarNxNAdjImplR<5>>, MScalarSUN);
MODULE_REGISTER_TMP(PosSpaceRenorm1DSU6, TPosSpaceRenorm1D<ScalarNxNAdjImplR<6>>, MScalarSUN);


/******************************************************************************
 *                 TPosSpaceRenorm1D implementation                             *
 ******************************************************************************/
// constructor /////////////////////////////////////////////////////////////////
template <typename SImpl>
TPosSpaceRenorm1D<SImpl>::TPosSpaceRenorm1D(const std::string name)
: Module<PosSpaceRenorm1DPar>(name)
{}

// dependencies/products ///////////////////////////////////////////////////////
template <typename SImpl>
std::vector<std::string> TPosSpaceRenorm1D<SImpl>::getInput(void)
{
    std::vector<std::string> in;
    std::set<std::string> ops;

    for (auto &p : par().op)
    {
        ops.insert(p.first);
        ops.insert(p.second);
    }
    for (auto &o : ops)
    {
        in.push_back(o);
    }

    return in;
}

template <typename SImpl>
std::vector<std::string> TPosSpaceRenorm1D<SImpl>::getOutput(void)
{
    std::vector<std::string> out = {};

    return out;
}

// setup ///////////////////////////////////////////////////////////////////////
template <typename SImpl>
void TPosSpaceRenorm1D<SImpl>::setup(void)
{
    const unsigned int nd = env().getDim().size();

    mom_.resize(par().mom.size());
    for (unsigned int i = 0; i < mom_.size(); ++i)
    {
        mom_[i] = strToVec<int>(par().mom[i]);
        if (mom_[i].size() != nd - 1)
        {
            HADRONS_ERROR(Size, "momentum number of components different from " + std::to_string(nd - 1));
        }
        for (unsigned int j = 0; j < nd - 1; ++j)
        {
            mom_[i][j] = (mom_[i][j] + env().getDim(j)) % env().getDim(j);
        }
    }
    envTmpLat(ComplexField, "ftBuf");
    envTmpLat(ComplexField, "opShiftBuf");
}

// execution ///////////////////////////////////////////////////////////////////
template <typename SImpl>
void TPosSpaceRenorm1D<SImpl>::execute(void)
{
    LOG(Message) << "Computing 2-point functions" << std::endl;
    for (auto &p : par().op)
    {
        LOG(Message) << "  <" << p.first << " " << p.second << ">" << std::endl;
    }

    const unsigned int nd = env().getNd();
    const unsigned int nt = env().getDim().back();
    const unsigned int nop = par().op.size();
    const unsigned int nmom = mom_.size();
    const double    zeroval = par().zeroval;
    const double  windowmin = par().windowmin;
    const double  windowmax = par().windowmax;
    double partVol = 1.;
    std::vector<int> dMask(nd, 1);
    std::set<std::string> ops;
    std::vector<PosSpaceRenorm1DResult> result;
    std::map<std::string, std::vector<SlicedOp>> slicedOp;
    FFT fft(envGetGrid(Field));
    TComplex buf;

    envGetTmp(ComplexField, ftBuf);
    envGetTmp(ComplexField, opShiftBuf);
    dMask[nd - 1] = 0;
    for (unsigned int mu = 0; mu < nd - 1; ++mu)
    {
        partVol *= env().getDim()[mu];
    }
    for (auto &p : par().op)
    {
        ops.insert(p.first);
        ops.insert(p.second);
    }
    for (auto &o : ops)
    {
        auto &op = envGet(ComplexField, o);
        Complex sum_op = TensorRemove(sum(op)) / static_cast<double>(nt * nt * nt);
        //opShiftBuf = op - sum_op;
        opShiftBuf = op;
        slicedOp[o].resize(nmom);
        LOG(Message) << "Operator '" << o << "' FFT" << std::endl;
        fft.FFT_dim_mask(ftBuf, opShiftBuf, dMask, FFT::forward);
        //fft.FFT_all_dim(ftBuf, op, FFT::forward);
        for (unsigned int m = 0; m < nmom; ++m)
        {
            auto qt = mom_[m];

            qt.resize(nd);
            slicedOp[o][m].resize(nt);
            for (unsigned int t = 0; t < nt; ++t)
            {
                qt[nd - 1] = t;
                peekSite(buf, ftBuf, qt);
                slicedOp[o][m][t] = TensorRemove(buf);
            }
        }
    }
    LOG(Message) << "Making contractions" << std::endl;
    for (unsigned int m = 0; m < nmom; ++m)
        for (auto &p : par().op)
        {
            PosSpaceRenorm1DResult r;

            r.sink = p.first;
            r.source = p.second;
            r.mom = mom_[m];
            r.data = makeTwoPointPosSpace1D(slicedOp[p.first][m], slicedOp[p.second][m],
                                            1. / partVol, zeroval, windowmin, windowmax);
            result.push_back(r);
        }
    saveResult(par().output, "twopt", result);
}

END_MODULE_NAMESPACE

END_HADRONS_NAMESPACE

#endif // Hadrons_MScalarSUN_PosSpaceRenorm1D_hpp_
