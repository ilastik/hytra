log2file() {
	  LOGFILE="$1" ; shift
	  { "$@" 2>&1 ; echo $? >"/tmp/~pipestatus.$$" ; } | tee -a "$LOGFILE"
	  MYPIPESTATUS="`cat \"/tmp/~pipestatus.$$\"`"
	  rm -f "/tmp/~pipestatus.$$"
	  return $MYPIPESTATUS
}

DATE=2015-10-21

OUT_DIR=/home/chaubold/data/iccv15/drosophila-old/result-cvpr16/m3-${DATE}-dynprog-update-swap
mkdir ${OUT_DIR}
log2file ${OUT_DIR}/log.txt python /home/chaubold/miniconda-2.7/envs/magnusson/src/embryonic/toolbox/multitrack_ilastik10 --load-traxelstore /home/chaubold/software/embryonic/toolbox/drosophila-2015-03-28_11-12-39.dump --method=conservation-dynprog --max-number-objects=4 --min-size=4 --max-neighbor-distance=100 --division-threshold=0.1 --ep_gap=0.0 --div=40 --tr=33 --app=50 --dis=100 --trans-par=5 --border-width=5 --num-threads=1 -o ${OUT_DIR} /home/chaubold/data/iccv15/drosophila-old/conservationTracking_2013-08-23_cellcount-replaced.ilp
python compare_tracking.py --quietly /home/chaubold/data/iccv15/drosophila-old/manual_tracking_20130910_size_geq4/ ${OUT_DIR}/iter_0 > ${OUT_DIR}/iter_0/result.txt
rm ${OUT_DIR}/iter_0/*.h5

OUT_DIR=/home/chaubold/data/iccv15/drosophila-old/result-cvpr16/m3-${DATE}-dynprog-update-noswap
mkdir ${OUT_DIR}
log2file ${OUT_DIR}/log.txt python /home/chaubold/miniconda-2.7/envs/magnusson/src/embryonic/toolbox/multitrack_ilastik10 --load-traxelstore /home/chaubold/software/embryonic/toolbox/drosophila-2015-03-28_11-12-39.dump --method=conservation-dynprog --max-number-objects=4 --min-size=4 --max-neighbor-distance=100 --division-threshold=0.1 --ep_gap=0.0 --div=40 --tr=33 --app=50 --dis=100 --trans-par=5 --border-width=5 --without-swaps --num-threads=1 -o ${OUT_DIR} /home/chaubold/data/iccv15/drosophila-old/conservationTracking_2013-08-23_cellcount-replaced.ilp
python compare_tracking.py --quietly /home/chaubold/data/iccv15/drosophila-old/manual_tracking_20130910_size_geq4/ ${OUT_DIR}/iter_0 > ${OUT_DIR}/iter_0/result.txt
rm ${OUT_DIR}/iter_0/*.h5

OUT_DIR=/home/chaubold/data/iccv15/drosophila-old/result-cvpr16/m3-${DATE}-ilp
mkdir ${OUT_DIR}
log2file ${OUT_DIR}/log.txt python /home/chaubold/miniconda-2.7/envs/magnusson/src/embryonic/toolbox/multitrack_ilastik10 --load-traxelstore /home/chaubold/software/embryonic/toolbox/drosophila-2015-03-28_11-12-39.dump --method=conservation --max-number-objects=4 --min-size=4 --max-neighbor-distance=100 --division-threshold=0.1 --ep_gap=0.0 --div=40 --tr=33 --app=50 --dis=100 --trans-par=5 --border-width=5 --num-threads=1 -o ${OUT_DIR} /home/chaubold/data/iccv15/drosophila-old/conservationTracking_2013-08-23_cellcount-replaced.ilp
python compare_tracking.py --quietly /home/chaubold/data/iccv15/drosophila-old/manual_tracking_20130910_size_geq4/ ${OUT_DIR}/iter_0 > ${OUT_DIR}/iter_0/result.txt
rm ${OUT_DIR}/iter_0/*.h5

OUT_DIR=/home/chaubold/data/iccv15/drosophila-old/result-cvpr16/m3-${DATE}-twostage-update-noswap
mkdir ${OUT_DIR}
log2file ${OUT_DIR}/log.txt pythonpython /home/chaubold/miniconda-2.7/envs/magnusson/src/embryonic/toolbox/multitrack_ilastik10 --load-traxelstore /home/chaubold/software/embryonic/toolbox/drosophila-2015-03-28_11-12-39.dump --method=conservation-twostage --max-number-objects=4 --min-size=4 --max-neighbor-distance=100 --division-threshold=0.1 --ep_gap=0.0 --div=40 --tr=33 --app=50 --dis=100 --trans-par=5 --border-width=5 --without-swaps --num-threads=1 -o ${OUT_DIR} /home/chaubold/data/iccv15/drosophila-old/conservationTracking_2013-08-23_cellcount-replaced.ilp
python compare_tracking.py --quietly /home/chaubold/data/iccv15/drosophila-old/manual_tracking_20130910_size_geq4/ ${OUT_DIR}/iter_0 > ${OUT_DIR}/iter_0/result.txt
rm ${OUT_DIR}/iter_0/*.h5

OUT_DIR=/home/chaubold/data/iccv15/drosophila-old/result-cvpr16/m3-${DATE}-twostage-update-swap
mkdir ${OUT_DIR}
log2file ${OUT_DIR}/log.txt python /home/chaubold/miniconda-2.7/envs/magnusson/src/embryonic/toolbox/multitrack_ilastik10 --load-traxelstore /home/chaubold/software/embryonic/toolbox/drosophila-2015-03-28_11-12-39.dump --method=conservation-twostage --max-number-objects=4 --min-size=4 --max-neighbor-distance=100 --division-threshold=0.1 --ep_gap=0.0 --num-threads=1 --div=40 --tr=33 --app=50 --dis=100 --trans-par=5 --border-width=5 -o ${OUT_DIR} /home/chaubold/data/iccv15/drosophila-old/conservationTracking_2013-08-23_cellcount-replaced.ilp
python compare_tracking.py --quietly /home/chaubold/data/iccv15/drosophila-old/manual_tracking_20130910_size_geq4/ ${OUT_DIR}/iter_0 > ${OUT_DIR}/iter_0/result.txt
rm ${OUT_DIR}/iter_0/*.h5

for p in {500,1000,1500,2000,2500}
do
	OUT_DIR=/home/chaubold/data/iccv15/drosophila-old/result-cvpr16/m3-${DATE}-twostage-update-swap-maxpaths-${p}
	mkdir ${OUT_DIR}
	log2file ${OUT_DIR}/log.txt python /home/chaubold/miniconda-2.7/envs/magnusson/src/embryonic/toolbox/multitrack_ilastik10 --load-traxelstore /home/chaubold/software/embryonic/toolbox/drosophila-2015-03-28_11-12-39.dump --method=conservation-twostage --max-number-objects=4 --min-size=4 --max-neighbor-distance=100 --division-threshold=0.1 --ep_gap=0.0 --div=40 --tr=33 --app=50 --dis=100 --trans-par=5 --border-width=5 --num-threads=1 --max-num-paths=${p} -o ${OUT_DIR} /home/chaubold/data/iccv15/drosophila-old/conservationTracking_2013-08-23_cellcount-replaced.ilp
	python compare_tracking.py --quietly /home/chaubold/data/iccv15/drosophila-old/manual_tracking_20130910_size_geq4/ ${OUT_DIR}/iter_0 > ${OUT_DIR}/iter_0/result.txt
	rm ${OUT_DIR}/iter_0/*.h5
done

for p in {500,1000,1500,2000,2500}
do
	OUT_DIR=/home/chaubold/data/iccv15/drosophila-old/result-cvpr16/m3-${DATE}-twostage-update-noswap-maxpaths-${p}
	mkdir ${OUT_DIR}
	log2file ${OUT_DIR}/log.txt python /home/chaubold/miniconda-2.7/envs/magnusson/src/embryonic/toolbox/multitrack_ilastik10 --load-traxelstore /home/chaubold/software/embryonic/toolbox/drosophila-2015-03-28_11-12-39.dump --method=conservation-twostage --max-number-objects=4 --min-size=4 --max-neighbor-distance=100 --division-threshold=0.1 --ep_gap=0.0 --div=40 --tr=33 --app=50 --dis=100 --trans-par=5 --border-width=5 --num-threads=1 --max-num-paths=${p} --without-swaps -o ${OUT_DIR} /home/chaubold/data/iccv15/drosophila-old/conservationTracking_2013-08-23_cellcount-replaced.ilp
	python compare_tracking.py --quietly /home/chaubold/data/iccv15/drosophila-old/manual_tracking_20130910_size_geq4/ ${OUT_DIR}/iter_0 > ${OUT_DIR}/iter_0/result.txt
	rm ${OUT_DIR}/iter_0/*.h5
done

# collect all relevant numbers
PREFIX=/home/chaubold/data/iccv15/drosophila-old/result-cvpr16
echo "" > ${PREFIX}/${DATE}.log
for i in `ls ${PREFIX}/m3-${DATE}-*/iter_0/result.txt`
do 
	name=`echo $i | sed -e "s:${PREFIX}/::g" | sed -e 's:/iter_0/result.txt::g'`
	logFile=`echo $i | sed -e 's:iter_0/result.txt:log.txt:g'`
	echo ${name},`python extract_timings_and_energies.py $logFile`,`cat $i` >> ${PREFIX}/${DATE}.log
done