#ifndef SGFEATURES_HXX
#define SGFEATURES_HXX


#include "vigra/transformimage.hxx"
#include "vigra/labelimage.hxx"
#include "vigra/inspectimage.hxx"
#include "vigra/basicimage.hxx"
#include "vigra/stdimage.hxx"

#include "vigra/functorexpression.hxx"

#include "vigra/multi_array.hxx"
#include <vigra/multi_pointoperators.hxx>

#include <vigra/mathutil.hxx>


#include "ConfigFeatures.hxx"

#include "math.h"
#include "iostream"


#include "vigra/timing.hxx"

namespace SGF {
    using namespace vigra::functor;

    /**
     * Functor to calculate center coordinates in N dimensions.
     * The coordinates are passed as a TinyVector of size N.
     */
    template <class T, int N>
    class FindCenter
    {
    public:

        typedef vigra::TinyVector<T,N> argument_type;
        typedef vigra::TinyVector<T,N> result_type;

        unsigned int count;
        result_type sum_coordinates;

        FindCenter()
            : sum_coordinates((T)0), count(0)
        {}

        void reset()
        {
            count = 0;
            sum_coordinates = result_type(0);
        }

        inline void operator()(argument_type const & coord)
        {
            sum_coordinates += coord;
            count++;
        }

        inline void operator()(FindCenter const & other)
        {
            count += other.count;
            sum_coordinates += other.sum_coordinates;
        }

        inline result_type operator()() const
        {
            if(count != 0)
                return result_type(sum_coordinates / count);
            else
                return result_type(-1);
        }
    };

    template <class T, class MASKTYPE, int N>
    class FindCenterMask: public FindCenter<T,N>
    {
    public:
        typedef MASKTYPE mask_type;
        typedef vigra::TinyVector<T,N> argument_type;
        typedef vigra::TinyVector<T,N> result_type;

        inline void operator()(argument_type const & coord, mask_type const & m)
        {
            if((unsigned int)m != 0){
                FindCenter<T,N>::operator()(coord);
            }
        }

        inline result_type operator()()
        {
            return FindCenter<T,N>::operator ()();
        }
    };




    /**
     * Functor to calculate the irregularity of a region.
     */
    template <class T, int N>
    class FindIRGL
    {
    public:

        typedef vigra::TinyVector<T,N> argument_type;
        typedef double result_type;
        argument_type center;
        unsigned int size;
        result_type IRGL;

        FindIRGL()
            : IRGL(0), center((T)0), size(0)
        {}

        FindIRGL(argument_type center, unsigned int size)
            : IRGL(0), center(center), size(size)
        {}

        void reset()
        {
            IRGL = result_type(0);
            center = argument_type(0);
        }

        inline void operator()(argument_type const & coord)
        {
            if(size != 0){
                double temp_result = (1+std::pow(4.*M_PI/3., 1./3.)*(coord - center).magnitude())/size -1;

                if(temp_result > IRGL){
                    IRGL = temp_result;
                }
            }
        }

        inline void operator()(FindIRGL const & other)
        {
            center = other.center;
            size = other.size;
            IRGL = other.IRGL;
        }

        inline result_type operator()() const
        {
            return IRGL;
        }
    };


    template <class VALUETYPE, class MASKTYPE>
    class FindMinMaxMask: public vigra::FindMinMax<VALUETYPE>
    {
    public:
        typedef MASKTYPE mask_type;
        typedef VALUETYPE argument_type;

        inline void operator()(argument_type const & v, mask_type const & m)
        {
            if((unsigned int)m != 0){
                vigra::FindMinMax<VALUETYPE>::operator()(v);
            }
        }
    };



    template <class T, class Functor>
    inline void
    inspectVolumeWithCoordinates(vigra::MultiArray<3, T>  & vol,
                                 Functor & f)
    {
        typedef typename vigra::MultiArray<3, T>::traverser iter3_t;
        typedef typename iter3_t::next_type iter2_t;
        typedef typename iter2_t::next_type iter1_t;

        iter3_t i3_f = vol.traverser_begin();
        iter3_t i3_l = vol.traverser_end();
        unsigned int i = 0;
        // iterate over the third dimension
        for ( ; i3_f != i3_l; ++i3_f, i++){

            iter2_t i2_f = i3_f.begin ();
            iter2_t i2_l = i3_f.end ();
            unsigned int j = 0;

            // iterate over the second dimension
            for ( ; i2_f != i2_l; ++i2_f, j++){
                iter1_t i1_f = i2_f.begin ();
                iter1_t i1_l = i2_f.end ();
                unsigned int k = 0;

                // iterate over the first dimension
                for (; i1_f != i1_l; ++i1_f, k++){
                    // construct TinyVector with coordinates
                    vigra::TinyVector<unsigned int, 3> coord (k,j,i);
                    // call functor
                    f(coord,*i1_f);
                }
            }
        }
    }


    template <class T>
    feature_array statistics(vigra::MultiArray<1,T> features)
    {
        int len = features.size(0);

        // return values:
        // [0] max value; [1] average value; [2] sample mean; [3] sample stddev
        feature_array ret (array_shape(4),-1.);

        if(len <= 1){
            return ret;
        }

        // calculate statistics by summation
        feature_type max_value  = features[0];
        feature_type sum_feat   = 0.;
        feature_type sum_t_feat = 0.;
        feature_type sum_t_sqr  = 0.;

        for(int t = 0; t < len; t++){
            if(features[t]>max_value)
                max_value = features[t] ;
            sum_feat   += features[t];
            sum_t_feat += t * features[t];
        }

        ret[0] = max_value;
        ret[1] = sum_feat / (len - 1);

        if(sum_feat == 0){
            return ret;
        }

        ret[2] = sum_t_feat / sum_feat;

        for(int t = 0; t < len; t++){
            sum_t_sqr += std::pow(t-ret[2],2) * features[t];
        }

        if(sum_t_sqr < 0){
            return ret;
        }
        ret[3] = std::sqrt( sum_t_sqr / sum_feat );


        return ret;

    }


    template <class T>
    feature_array SGFeatures(vigra::MultiArray<3,T> src_volume, unsigned int levels = 16, T background = 0)
    {

        typedef bool mask_type;
        typedef T value_type;

        // create a mask volume
        vigra::MultiArray<3, mask_type> mask_volume (src_volume.shape());
        vigra::transformMultiArray(vigra::srcMultiArrayRange(src_volume),
                                   vigra::destMultiArray(mask_volume),
                                   vigra::functor::ifThenElse(
                                           Arg1() == Param(background),
                                           Param(0),
                                           Param(1)
                                           )
                                   );




        // get minimum/maximum values
        FindMinMaxMask<value_type,mask_type> minmax;
        vigra::inspectTwoMultiArrays(vigra::srcMultiArrayRange(src_volume),
                                     vigra::srcMultiArray(mask_volume),
                                     minmax
                                     );

        value_type maxint = minmax.max;
        value_type minint = minmax.min;
        value_type intrange = maxint - minint;


        // Storage for the features
        feature_array NCA[2]     = {feature_array(array_shape(levels)),
                                    feature_array(array_shape(levels))};
        feature_array IRGL[2]    = {feature_array(array_shape(levels)),
                                    feature_array(array_shape(levels))};
        feature_array DISP[2]    = {feature_array(array_shape(levels)),
                                    feature_array(array_shape(levels))};
        feature_array INERTIA[2] = {feature_array(array_shape(levels)),
                                    feature_array(array_shape(levels))};
        feature_array TAREA[2]   = {feature_array(array_shape(levels)),
                                    feature_array(array_shape(levels))};
        feature_array CAREA[2]   = {feature_array(array_shape(levels)),
                                    feature_array(array_shape(levels))};

        // apply all thresholds
        for (int t = 1; t < levels; t++)
        {
            label_volume thr_volume (volume_shape(src_volume.shape()));

            value_type threshold = minint + value_type(double(intrange) / double(levels) * t);

            // thresholding
            vigra::transformMultiArray(vigra::srcMultiArrayRange(src_volume),
                                       vigra::destMultiArray(thr_volume),
                                       vigra::functor::ifThenElse(
                                               Arg1() < Param(threshold),
                                               Param(0),
                                               Param(1)
                                               )
                                       );


            unsigned int label_count[2];
            label_volume lbl_volume [2] =
            {label_volume(volume_shape(src_volume.shape())),
             label_volume(volume_shape(src_volume.shape()))};

            // first label only the 1-regions (higher than threshold)
            label_count[1] = vigra::labelVolumeWithBackground(
                    vigra::srcMultiArrayRange(thr_volume),
                    vigra::destMultiArray(lbl_volume[1]),
                    vigra::NeighborCode3DSix(),0);

            // relabel all _masked_ threshold labels: 0-->1, 1-->0
            // cant use functors here, transformMultiArrayIf is not implemented
            for (int a = 0 ; a < thr_volume.shape(0); a++){
                for (int b = 0 ; b < thr_volume.shape(1); b++){
                    for (int c = 0 ; c < thr_volume.shape(2); c++){
                        if(mask_volume(a,b,c) != 0){
                            if(thr_volume(a,b,c) == 0){
                                thr_volume(a,b,c) = 1;
                            }else{
                                thr_volume(a,b,c) = 0;
                            }
                        }
                    }
                }
            }


            // label 1-regions (now lower than threshold)
            label_count[0] = vigra::labelVolumeWithBackground(
                    vigra::srcMultiArrayRange(thr_volume),
                    vigra::destMultiArray(lbl_volume[0]),
                    vigra::NeighborCode3DSix(),0);


            // Get the Center Of Gravity for each region
            typedef vigra::ArrayOfRegionStatistics< FindCenter<label_type,3> > f_center;
            f_center COG[2] = {f_center(label_count[0]), f_center(label_count[1])};

            // Get the size of each region
            typedef vigra::ArrayOfRegionStatistics<
                    vigra::FindROISize<label_type> > f_size;
            f_size size[2] = {f_size(label_count[0]),f_size(label_count[1])};

            // Get the irregularity for each region
            typedef vigra::ArrayOfRegionStatistics< FindIRGL<label_type,3> > f_irgl;
            f_irgl IRGL_j[2] = {f_irgl(label_count[0]), f_irgl(label_count[1])};

            // Go through volume and collect the above values
            for (int i = 0; i < 2; ++i)
            {
                inspectVolumeWithCoordinates(lbl_volume[i], COG[i]);

                vigra::inspectTwoMultiArrays(srcMultiArrayRange(lbl_volume[i]),
                                             srcMultiArray(lbl_volume[i]), size[i]);

                // Init irregularity functor with center and size of each clump
                for(int j = 1; j <= label_count[i]; j++){
                    IRGL_j[i][j]( FindIRGL<label_type,3> (COG[i][j](), size[i][j]()) );
                }
                inspectVolumeWithCoordinates(lbl_volume[i], IRGL_j[i]);
            }

            // Calculate the Center Of Gravity of the Nucleus
            FindCenterMask<label_type,mask_type,3> COGN;
            inspectVolumeWithCoordinates(mask_volume,COGN);

            double total_size = COGN.count;
            double sqrt_total_size = std::sqrt(total_size);
            double sqrt_pi = std::sqrt(M_PI);


            // Now calculate the actual feature values:
            //
            // NCA -- Normalised Number of Connected Regions
            // IRGL -- Irregularity
            // DISP -- Average Clump Displacement
            // INERTIA -- Average Clump Interia
            // TAREA -- Total Clump Area
            // CAREA -- Average Clump Area
            for (int i = 0; i < 2; ++i)
            {
                double sum_Dj = 0.;       // normalized clump displacement
                double sum_Dj_NOPj = 0.;  // D_j * NOP_j for average Inertia
                double sum_NOPj = 0.;     // area calculations
                double sum_IRGLj = 0.;    // irregularity

                for (int j = 1; j <= label_count[i]; ++j)
                {
                    double Dj = ( sqrt_pi * (COG[i][j]() - COGN()).magnitude() / sqrt_total_size);
                    sum_Dj += Dj;
                    sum_Dj_NOPj += Dj * size[i][j]();
                    sum_NOPj += size[i][j]();
                    sum_IRGLj += IRGL_j[i][j]();
                }
                NCA[i][t] = label_count[i] / total_size;
                IRGL[i][t] = sum_IRGLj;
                TAREA[i][t] = sum_NOPj / total_size;
                if(label_count[i] > 0){
                    DISP[i][t] = sum_Dj / label_count[i];
                    INERTIA[i][t] = sum_Dj_NOPj / label_count[i];
                    CAREA[i][t] = sum_NOPj / label_count[i];
                }

            }

        } // END loop over t

        // calculate statistics over all threshold levels
        feature_array STAT_NCA[2]     = {feature_array(array_shape(4)),
                                         feature_array(array_shape(4))};
        feature_array STAT_IRGL[2]    = {feature_array(array_shape(4)),
                                         feature_array(array_shape(4))};
        feature_array STAT_DISP[2]    = {feature_array(array_shape(4)),
                                         feature_array(array_shape(4))};
        feature_array STAT_INERTIA[2] = {feature_array(array_shape(4)),
                                         feature_array(array_shape(4))};
        feature_array STAT_TAREA[2]   = {feature_array(array_shape(4)),
                                         feature_array(array_shape(4))};
        feature_array STAT_CAREA[2]   = {feature_array(array_shape(4)),
                                         feature_array(array_shape(4))};

        for(int i = 0;  i < 2; i++){
            STAT_NCA[i]    = statistics(NCA[i]);
            STAT_IRGL[i]   = statistics(IRGL[i]);
            STAT_DISP[i]   = statistics(DISP[i]);
            STAT_INERTIA[i]= statistics(INERTIA[i]);
            STAT_TAREA[i]  = statistics(TAREA[i]);
            STAT_CAREA[i]  = statistics(CAREA[i]);
        }


        // fill values into output vector
        feature_array ret (array_shape(48));
        for(int i = 0; i < 2; i++){
            int o = i*24;
            for(int j = 0; j < 4; j++){
                ret[o + j]      = STAT_NCA[i][j];
                ret[o + 4 + j]  = STAT_IRGL[i][j];
                ret[o + 8 + j]  = STAT_DISP[i][j];
                ret[o + 12 + j] = STAT_INERTIA[i][j];
                ret[o + 16 + j] = STAT_TAREA[i][j];
                ret[o + 20 + j] = STAT_CAREA[i][j];
            }
        }

        return ret;
    }



}; // END namespace SGF



#endif // SGFEATURES_HXX
