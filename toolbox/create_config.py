import argparse
import jinja2
import itertools

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Generate a tracking pipeline config file by instantiating a template.

        Example: 
            python create_config.py --in config_template_ctc.ini --out testconfig.ini \
            outDir /Users/chaubold/hci/projects/cell_tracking_challenge_15/Fluo-N2DH-SIM/01_pipeline-2016-04-29  \
            datasetDir /Users/chaubold/hci/projects/cell_tracking_challenge_15/Fluo-N2DH-SIM  \
            sequence 01  \
            ilp /Users/chaubold/hci/projects/cell_tracking_challenge_15/Fluo-N2DH-SIM/01/tracking-2015-02-27.ilp  \
            numFrames 55  \
            trackingExecutable /Users/chaubold/opt/miniconda/envs/virginie/src/multiHypoTracking/build/bin/track
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--in', dest='template_filename', required=True, type=str,
                    help='Template filename')
    parser.add_argument('--out', dest='config_filename', required=True, type=str,
                    help='Resulting config filename')

    options, unknown = parser.parse_known_args()

    assert(len(unknown) % 2 == 0)
    dictParameters = zip(unknown[0::2], unknown[1::2])

    with open(options.template_filename, 'r') as f:
        template_string = f.read()
        template = jinja2.Template(template_string)
        config_string = template.render(dictParameters)
        with open(options.config_filename, 'w') as out:
            out.write(config_string)
    