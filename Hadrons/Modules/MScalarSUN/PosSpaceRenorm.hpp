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
                                    unsigned int,              samp,
                                    double,                    windowmin,
                                    double,                    windowmax,
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
    typedef Grid::iVector<Grid::RealD, 2000> Vec;
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

    const uint64_t                    nsimd         = vComplex::Nsimd();
    const unsigned int                nd            = env().getDim().size();
    const unsigned int                L             = env().getDim().back(); 
    const unsigned int                nop           = par().op.size();
    //const unsigned int              samp          = par().samp;
    unsigned int                      samp          = par().samp;
    double                            windowmin     = par().windowmin;
    double                            windowmax     = par().windowmax;
    const unsigned int                nmom          = mom_.size();
    std::vector<PosSpaceRenormResult> result;
    std::set<std::string>             ops;
    std::vector<int>                  shift(nd,0);
    FFT                               fft(envGetGrid(Field));
    std::vector<Complex>              res(L, 0.);
    TComplex                          TCbuf1,    TCbuf2;
    Vec                               rn;
    GridSerialRNG                     sRNG;   
    sRNG.SeedFixedIntegers(std::vector<int>({45,12,81}));
    random(sRNG, rn);

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
    windowFuncField = sqrt(windowFuncField);
    for(int i = 0; i < L; i++)
    {
        for(int j = 0; j < L; j++)
        {
            for(int k = 0; k < L; k++)
            {
                shift = {i, j, k};
                peekSite(TCbuf1, windowFuncField, shift);
                TCbuf1 = windowFunction(windowmin, windowmax, TensorRemove(TCbuf1).real(), 1000);
                pokeSite(TCbuf1, windowFuncField, shift);
            }
        }
        
    }

    shift = {0, 0, 0};
    
    for (unsigned int m = 0; m < nmom; ++m)
    for (auto &p: par().op)
    {
        auto &op1 = envGet(ComplexField, p.first);
        auto &op2 = envGet(ComplexField, p.second);
        auto qt = mom_[m];
        qt.resize(nd);

        //samp = L*L*L; //temp
        for(int i = 0; i < samp; i++)
        {
            ft_buf_in = op2;
            //shift[0] = i%L; //temp
            //shift[1] = (i/L)%L;//temp
            //shift[2] = (i/L/L);//temp
            if(samp != 1){
                for(int mu = 0; mu < nd; mu++)
                {
                    shift[mu] = rn((i+mu)%samp)*L; //to restore from temp
                    ft_buf_in = Cshift(ft_buf_in, mu, shift[mu]);
                } 
            }
            LOG(Message) << "shift coordinate = " << shift << std::endl;
            peekSite(TCbuf1, op1, shift);
            //ft_buf_in = ft_buf_in * windowFuncField; //don't multiply window function
            fft.FFT_all_dim(ft_buf_out, ft_buf_in, FFT::forward);
            for (int t = 0; t < L; t++)
            {
                qt[nd-1] = t;
                peekSite(TCbuf2, adj(ft_buf_out), qt);
                res[t] += trace(TensorRemove(TCbuf1*TCbuf2))/double(samp); // divide some volume factors?
            }
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
        PosSpaceRenormResult r;

        r.sink   = p.first;
        r.source = p.second;
        r.mom    = mom_[m];
        r.data   = res;
        result.push_back(r);
    }
    saveResult(par().output, "twopt", result);
}

END_MODULE_NAMESPACE

END_HADRONS_NAMESPACE

#endif // Hadrons_MScalarSUN_PosSpaceRenorm_hpp_
