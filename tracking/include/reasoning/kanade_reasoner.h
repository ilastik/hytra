#ifndef KANADE_REASONER_H
#define KANADE_REASONER_H

#include "reasoning/reasoner.h"

namespace Tracking {
class Traxel;
class HypothesesGraph;

class Kanade : public Reasoner {
    public:
    virtual void formulate( const HypothesesGraph& );
    virtual void infer();
    virtual void conclude( HypothesesGraph& );

    private:
    // copy and assingment have to be implemented, yet
    //SingleTimestepTraxelMrf(const SingleTimestepTraxelMrf&) {};
    //SingleTimestepTraxelMrf& operator=(const SingleTimestepTraxelMrf&) { return *this;};
};

} /* namespace Tracking */
#endif /* KANADE_REASONER_H */
