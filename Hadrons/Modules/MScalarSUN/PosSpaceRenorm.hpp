#ifndef Hadrons_MScalarSUN_PosSpaceRenorm_hpp_
#define Hadrons_MScalarSUN_PosSpaceRenorm_hpp_

#include <Hadrons/Global.hpp>
#include <Hadrons/Module.hpp>
#include <Hadrons/ModuleFactory.hpp>
#include <Hadrons/Modules/MScalarSUN/Utils.hpp>

BEGIN_HADRONS_NAMESPACE

/******************************************************************************
 *                         PosSpaceRenorm                                 *
 ******************************************************************************/
BEGIN_MODULE_NAMESPACE(MScalarSUN)

class PosSpaceRenormPar: Serializable
{
public:
    typedef std::pair<std::string, std::string> OpPair;
    GRID_SERIALIZABLE_CLASS_MEMBERS(PosSpaceRenormPar,
                                    std::vector<OpPair>,       op,
                                    std::vector<std::string>,  mom,
                                    std::string,               output);
};

class PosSpaceRenormResult: Serializable
{
public:
    GRID_SERIALIZABLE_CLASS_MEMBERS(PosSpaceRenormResult,
                                    std::string, sink,
                                    std::string, source,
                                    std::vector<int>, mom,
                                    std::vector<Complex>, data);
};

template <typename SImpl>
class TPosSpaceRenorm: public Module<PosSpaceRenormPar>
{
public:
    typedef typename SImpl::Field         Field;
    typedef typename SImpl::ComplexField  ComplexField;
public:
    // constructor
    TPosSpaceRenorm(const std::string name);
    // destructor
    virtual ~TPosSpaceRenorm(void) {};
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

MODULE_REGISTER_TMP(PosSpaceRenormSU2, TPosSpaceRenorm<ScalarNxNAdjImplR<2>>, MScalarSUN);
MODULE_REGISTER_TMP(PosSpaceRenormSU3, TPosSpaceRenorm<ScalarNxNAdjImplR<3>>, MScalarSUN);
MODULE_REGISTER_TMP(PosSpaceRenormSU4, TPosSpaceRenorm<ScalarNxNAdjImplR<4>>, MScalarSUN);
MODULE_REGISTER_TMP(PosSpaceRenormSU5, TPosSpaceRenorm<ScalarNxNAdjImplR<5>>, MScalarSUN);
MODULE_REGISTER_TMP(PosSpaceRenormSU6, TPosSpaceRenorm<ScalarNxNAdjImplR<6>>, MScalarSUN);


/******************************************************************************
 *                 TPosSpaceRenorm implementation                             *
 ******************************************************************************/
// constructor /////////////////////////////////////////////////////////////////
template <typename SImpl>
TPosSpaceRenorm<SImpl>::TPosSpaceRenorm(const std::string name)
: Module<PosSpaceRenormPar>(name)
{}

// dependencies/products ///////////////////////////////////////////////////////
template <typename SImpl>
std::vector<std::string> TPosSpaceRenorm<SImpl>::getInput(void)
{
    std::vector<std::string> in;
    std::set<std::string>    ops;

    for (auto &p: par().op)
    {
        ops.insert(p.first);
        ops.insert(p.second);
    }
    for (auto &o: ops)
    {
        in.push_back(o);
    }

    return in;
}

template <typename SImpl>
std::vector<std::string> TPosSpaceRenorm<SImpl>::getOutput(void)
{
    std::vector<std::string> out = {};
    
    return out;
}

// setup ///////////////////////////////////////////////////////////////////////
template <typename SImpl>
void TPosSpaceRenorm<SImpl>::setup(void)
{
    const unsigned int nd = env().getDim().size();

    mom_.resize(par().mom.size());
    for (unsigned int i = 0; i < mom_.size(); ++i)
    {
        mom_[i] = strToVec<int>(par().mom[i]);
        if (mom_[i].size() != nd - 1)
        {
            HADRONS_ERROR(Size, "momentum number of components different from " 
                               + std::to_string(nd-1));
        }
        for (unsigned int j = 0; j < nd - 1; ++j)
        {
            mom_[i][j] = (mom_[i][j] + env().getDim(j)) % env().getDim(j);
        }
    }
    
    envTmpLat(ComplexField, "coor");
    envTmpLat(ComplexField, "windowFuncField");
    envTmpLat(ComplexField, "ft_buf_in");
    envTmpLat(ComplexField, "ft_buf_out");
}

// execution ///////////////////////////////////////////////////////////////////
template <typename SImpl>
void TPosSpaceRenorm<SImpl>::execute(void)
{

    const uint64_t                    nsimd   = vComplex::Nsimd();
    const unsigned int                nd      = env().getDim().size();
    const unsigned int                nop     = par().op.size();
    std::set<std::string>             ops;
    int                               Nsite;
    std::vector<int>                  qt(3,0), shift(3,0);
    const unsigned int                L       = env().getDim().back(); // ????
    FFT                               fft(envGetGrid(Field));
    std::vector<int>                  dMask(nd, 1);
    std::vector<Complex>              res(L, 0.);

    envGetTmp(ComplexField, coor);
    envGetTmp(ComplexField, windowFuncField);
    envGetTmp(ComplexField, ft_buf_in);
    envGetTmp(ComplexField, ft_buf_out);

    for (auto &p: par().op)
    {
        ops.insert(p.first);
        ops.insert(p.second);
    }

    windowFuncField = 0.0;
    for(int mu = 0; mu < nd; mu++) 
    {
        LatticeCoordinate(coor,mu);
        coor = (L-abs(Integer(2)*coor-L))*RealD(0.5);
        windowFuncField += coor*coor;
    }
    
    TComplex TCbuf1, TCbuf2; 
    for (int i = 0; i < L; i++)
    {
        qt = {0, 0, i};
        peekSite(TCbuf1, windowFuncField, qt);
        //LOG(Message) << "qt = " << qt << "    coor = " << TCbuf1 << std::endl;
        qt = {0, i, i};
        peekSite(TCbuf1, windowFuncField, qt);
        //LOG(Message) << "qt = " << qt << "    coor = " << TCbuf1 << std::endl;

    }
    windowFuncField = 1.0;
    int mu;
    for (auto &p: par().op)
    {
        auto &op1 = envGet(ComplexField, p.first);
        auto &op2 = envGet(ComplexField, p.second);

        for(int i = 0; i < L; i++) //Change back to L
        {
            for (int j = 0; j < L; j++) //Change back to L
            {
                for (int k = 0; k < L; k++) //Change back to L
                {
                    
                    shift = {i,j,k};
                    peekSite(TCbuf1, op1, shift);
                    ft_buf_in = adj(op2) * windowFuncField;
                    fft.FFT_all_dim(ft_buf_out, ft_buf_in, FFT::forward);

                    for (int t = 0; t < L; t++)
                    {
                        qt = {1, 1, t};
                        peekSite(TCbuf2, ft_buf_out, qt);
                        res[t] += TensorRemove(TCbuf1)*TensorRemove(TCbuf2);
                    }
                    mu = 2;
                    op2 = Cshift(op2, mu, 1);
                    
                }
                mu = 1;
                op2 = Cshift(op2, mu, 1);
                LOG(Message) << "still going i =  " << i << "  j = " << j << std::endl;
            } 
            mu = 0;
            op2 = Cshift(op2, mu, 1);
            
        }

        //Experimental

       /*  autoView(op1_v, op1, AcceleratorRead);
        op1.Grid()->show_decomposition();

        autoView(op2_v, op2, AcceleratorRead);
        autoView(ft_buf_in_v, ft_buf_in, AcceleratorRead);
        autoView(ft_buf_out_v, ft_buf_out, AcceleratorWrite);


        accelerator_for(ss, op1.Grid()->gSites(), nsimd,
                        {
                            auto temp = coalescedRead(op1_v[ss]);
                            LOG(Message) << "ss:  " << ss << "  temp: " << temp << std::endl;

                        });

        LOG(Message) << "op1 size " << op1_v.size() << std::endl; */
    }


}

END_MODULE_NAMESPACE

END_HADRONS_NAMESPACE

#endif // Hadrons_MScalarSUN_PosSpaceRenorm_hpp_
