
-------------------
Command line usage
-------------------



.. raw:: html

  <div class="highlight-default"><div class="highlight"><pre><span></span><span class="n">usage</span><span class="p">:</span> <span class="n">neuropredict</span> <span class="p">[</span><span class="o">-</span><span class="n">h</span><span class="p">]</span> <span class="p">[</span><span class="o">-</span><span class="n">m</span> <span class="n">META_FILE</span><span class="p">]</span> <span class="p">[</span><span class="o">-</span><span class="n">o</span> <span class="n">OUT_DIR</span><span class="p">]</span> <span class="p">[</span><span class="o">-</span><span class="n">f</span> <span class="n">FS_SUBJECT_DIR</span><span class="p">]</span>
                      <span class="p">[</span><span class="o">-</span><span class="n">y</span> <span class="n">PYRADIGM_PATHS</span> <span class="p">[</span><span class="n">PYRADIGM_PATHS</span> <span class="o">...</span><span class="p">]]</span>
                      <span class="p">[</span><span class="o">-</span><span class="n">u</span> <span class="n">USER_FEATURE_PATHS</span> <span class="p">[</span><span class="n">USER_FEATURE_PATHS</span> <span class="o">...</span><span class="p">]]</span>
                      <span class="p">[</span><span class="o">-</span><span class="n">d</span> <span class="n">DATA_MATRIX_PATHS</span> <span class="p">[</span><span class="n">DATA_MATRIX_PATHS</span> <span class="o">...</span><span class="p">]]</span>
                      <span class="p">[</span><span class="o">-</span><span class="n">a</span> <span class="n">ARFF_PATHS</span> <span class="p">[</span><span class="n">ARFF_PATHS</span> <span class="o">...</span><span class="p">]]</span> <span class="p">[</span><span class="o">-</span><span class="n">p</span> <span class="n">POSITIVE_CLASS</span><span class="p">]</span>
                      <span class="p">[</span><span class="o">-</span><span class="n">t</span> <span class="n">TRAIN_PERC</span><span class="p">]</span> <span class="p">[</span><span class="o">-</span><span class="n">n</span> <span class="n">NUM_REP_CV</span><span class="p">]</span>
                      <span class="p">[</span><span class="o">-</span><span class="n">k</span> <span class="n">NUM_FEATURES_TO_SELECT</span><span class="p">]</span>
                      <span class="p">[</span><span class="o">-</span><span class="n">s</span> <span class="p">[</span><span class="n">SUB_GROUPS</span> <span class="p">[</span><span class="n">SUB_GROUPS</span> <span class="o">...</span><span class="p">]]]</span>
                      <span class="p">[</span><span class="o">-</span><span class="n">g</span> <span class="p">{</span><span class="n">none</span><span class="p">,</span><span class="n">light</span><span class="p">,</span><span class="n">exhaustive</span><span class="p">}]</span>
                      <span class="p">[</span><span class="o">-</span><span class="n">fs</span> <span class="p">{</span><span class="n">selectkbest_mutual_info_classif</span><span class="p">,</span><span class="n">selectkbest_f_classif</span><span class="p">,</span><span class="n">variancethreshold</span><span class="p">}]</span>
                      <span class="p">[</span><span class="o">-</span><span class="n">e</span> <span class="p">{</span><span class="n">randomforestclassifier</span><span class="p">,</span><span class="n">extratreesclassifier</span><span class="p">,</span><span class="n">svm</span><span class="p">}]</span>
                      <span class="p">[</span><span class="o">-</span><span class="n">z</span> <span class="n">MAKE_VIS</span><span class="p">]</span> <span class="p">[</span><span class="o">-</span><span class="n">c</span> <span class="n">NUM_PROCS</span><span class="p">]</span> <span class="p">[</span><span class="o">-</span><span class="n">v</span><span class="p">]</span>
  </pre></div>
  </div>
  <div class="section" id="Named Arguments">
  <h2>Named Arguments<a class="headerlink" href="#Named Arguments" title="Permalink to this headline">¶</a></h2>
  <table class="docutils option-list" frame="void" rules="none">
  <col class="option" />
  <col class="description" />
  <tbody valign="top">
  <tr><td class="option-group" colspan="2">
  <kbd>-m, --meta_file</kbd></td>
  </tr>
  <tr><td>&#160;</td><td><p class="first">Abs path to file containing metadata for subjects to be included for analysis.</p>
  <p>At the minimum, each subject should have an id per row followed by the class it belongs to.</p>
  <p>E.g.
  .. parsed-literal:</p>
  <div class="last highlight-default"><div class="highlight"><pre><span></span><span class="n">sub001</span><span class="p">,</span><span class="n">control</span>
  <span class="n">sub002</span><span class="p">,</span><span class="n">control</span>
  <span class="n">sub003</span><span class="p">,</span><span class="n">disease</span>
  <span class="n">sub004</span><span class="p">,</span><span class="n">disease</span>
  </pre></div>
  </div>
  </td></tr>
  <tr><td class="option-group">
  <kbd>-o, --out_dir</kbd></td>
  <td>Output folder to store gathered features &amp; results.</td></tr>
  <tr><td class="option-group" colspan="2">
  <kbd>-f, --fs_subject_dir</kbd></td>
  </tr>
  <tr><td>&#160;</td><td><p class="first">Absolute path to <code class="docutils literal"><span class="pre">SUBJECTS_DIR</span></code> containing the finished runs of Freesurfer parcellation
  Each subject will be queried after its ID in the metadata file.</p>
  <p class="last">E.g. <code class="docutils literal"><span class="pre">--fs_subject_dir</span> <span class="pre">/project/freesurfer_v5.3</span></code></p>
  </td></tr>
  </tbody>
  </table>
  </div>
  <div class="section" id="Input data and formats">
  <h2>Input data and formats<a class="headerlink" href="#Input data and formats" title="Permalink to this headline">¶</a></h2>
  <p>Only one of the following types can be specified.</p>
  <table class="docutils option-list" frame="void" rules="none">
  <col class="option" />
  <col class="description" />
  <tbody valign="top">
  <tr><td class="option-group" colspan="2">
  <kbd>-y, --pyradigm_paths</kbd></td>
  </tr>
  <tr><td>&#160;</td><td><p class="first">Path(s) to pyradigm datasets.</p>
  <p class="last">Each path is self-contained dataset identifying each sample, its class and features.</p>
  </td></tr>
  <tr><td class="option-group" colspan="2">
  <kbd>-u, --user_feature_paths</kbd></td>
  </tr>
  <tr><td>&#160;</td><td><p class="first">List of absolute paths to user’s own features.</p>
  <p>Format: Each of these folders contains a separate folder for each subject (named after its ID in the metadata file)
  containing a file called features.txt with one number per line.
  All the subjects (in a given folder) must have the number of features (#lines in file).
  Different parent folders (describing one feature set) can have different number of features for each subject,
  but they must all have the same number of subjects (folders) within them.</p>
  <p>Names of each folder is used to annotate the results in visualizations.
  Hence name them uniquely and meaningfully, keeping in mind these figures will be included in your papers.
  For example,</p>
  <div class="highlight-default"><div class="highlight"><pre><span></span><span class="o">--</span><span class="n">user_feature_paths</span> <span class="o">/</span><span class="n">project</span><span class="o">/</span><span class="n">fmri</span><span class="o">/</span> <span class="o">/</span><span class="n">project</span><span class="o">/</span><span class="n">dti</span><span class="o">/</span> <span class="o">/</span><span class="n">project</span><span class="o">/</span><span class="n">t1_volumes</span><span class="o">/</span>
  </pre></div>
  </div>
  <p class="last">Only one of <code class="docutils literal"><span class="pre">--pyradigm_paths</span></code>, <code class="docutils literal"><span class="pre">user_feature_paths</span></code>, <code class="docutils literal"><span class="pre">data_matrix_path</span></code> or <code class="docutils literal"><span class="pre">arff_paths</span></code> options can be specified.</p>
  </td></tr>
  <tr><td class="option-group" colspan="2">
  <kbd>-d, --data_matrix_paths</kbd></td>
  </tr>
  <tr><td>&#160;</td><td><p class="first">List of absolute paths to text files containing one matrix of size N x p  (num_samples x num_features).</p>
  <p>Each row in the data matrix file must represent data corresponding to sample in the same row
  of the meta data file (meta data file and data matrix must be in row-wise correspondence).</p>
  <p>Name of this file will be used to annotate the results and visualizations.</p>
  <p>E.g. <a href="#id1"><span class="problematic" id="id2">``</span></a>–data_matrix_paths /project/fmri.csv /project/dti.csv /project/t1_volumes.csv ``</p>
  <p>Only one of <code class="docutils literal"><span class="pre">--pyradigm_paths</span></code>, <code class="docutils literal"><span class="pre">user_feature_paths</span></code>, <code class="docutils literal"><span class="pre">data_matrix_path</span></code> or <code class="docutils literal"><span class="pre">arff_paths</span></code> options can be specified.</p>
  <dl class="last docutils">
  <dt>File format could be</dt>
  <dd><ul class="first simple">
  <li><dl class="first docutils">
  <dt>a simple comma-separated text file (with extension .csv or .txt): which can easily be read back with</dt>
  <dd>numpy.loadtxt(filepath, delimiter=’,’)
  or</dd>
  </dl>
  </li>
  <li>a numpy array saved to disk (with extension .npy or .numpy) that can read in with numpy.load(filepath).</li>
  </ul>
  <p>One could use <code class="docutils literal"><span class="pre">numpy.savetxt(data_array,</span> <span class="pre">delimiter=',')</span></code> or <code class="docutils literal"><span class="pre">numpy.save(data_array)</span></code> to save features.</p>
  <p class="last">File format is inferred from its extension.</p>
  </dd>
  </dl>
  </td></tr>
  <tr><td class="option-group" colspan="2">
  <kbd>-a, --arff_paths</kbd></td>
  </tr>
  <tr><td>&#160;</td><td><p class="first">List of paths to files saved in Weka’s ARFF dataset format.</p>
  <dl class="docutils">
  <dt>Note:</dt>
  <dd><ul class="first last simple">
  <li>this format does NOT allow IDs for each subject.</li>
  <li><dl class="first docutils">
  <dt>given feature values are saved in text format, this can lead to large files with high-dimensional data,</dt>
  <dd>compared to numpy arrays saved to disk in binary format.</dd>
  </dl>
  </li>
  </ul>
  </dd>
  </dl>
  <p class="last">More info: <a class="reference external" href="https://www.cs.waikato.ac.nz/ml/weka/arff.html">https://www.cs.waikato.ac.nz/ml/weka/arff.html</a></p>
  </td></tr>
  </tbody>
  </table>
  </div>
  <div class="section" id="Cross-validation">
  <h2>Cross-validation<a class="headerlink" href="#Cross-validation" title="Permalink to this headline">¶</a></h2>
  <p>Parameters related to training and optimization during cross-validation</p>
  <table class="docutils option-list" frame="void" rules="none">
  <col class="option" />
  <col class="description" />
  <tbody valign="top">
  <tr><td class="option-group" colspan="2">
  <kbd>-p, --positive_class</kbd></td>
  </tr>
  <tr><td>&#160;</td><td><p class="first">Name of the positive class (e.g. Alzheimers, MCI etc) to be used in calculation of area under the ROC curve.
  Applicable only for binary classification experiments.</p>
  <p class="last">Default: class appearing last in order specified in metadata file.</p>
  </td></tr>
  <tr><td class="option-group" colspan="2">
  <kbd>-t, --train_perc</kbd></td>
  </tr>
  <tr><td>&#160;</td><td><p class="first">Percentage of the smallest class to be reserved for training.</p>
  <p>Must be in the interval [0.01 0.99].</p>
  <p class="last">If sample size is sufficiently big, we recommend 0.5.
  If sample size is small, or class imbalance is high, choose 0.8.</p>
  </td></tr>
  <tr><td class="option-group" colspan="2">
  <kbd>-n, --num_rep_cv</kbd></td>
  </tr>
  <tr><td>&#160;</td><td><p class="first">Number of repetitions of the repeated-holdout cross-validation.</p>
  <p class="last">The larger the number, more stable the estimates will be.</p>
  </td></tr>
  <tr><td class="option-group" colspan="2">
  <kbd>-k, --num_features_to_select</kbd></td>
  </tr>
  <tr><td>&#160;</td><td><dl class="first last docutils">
  <dt>Number of features to select as part of feature selection.</dt>
  <dd><p class="first">Options:</p>
  <blockquote>
  <div><ul class="simple">
  <li>‘tenth’</li>
  <li>‘sqrt’</li>
  <li>‘log2’</li>
  <li>‘all’</li>
  </ul>
  </div></blockquote>
  <p>Default: ‘tenth’ of the number of samples in the training set.</p>
  <p class="last">For example, if your dataset has 90 samples, you chose 50 percent for training (default),
  then Y will have 90*.5=45 samples in training set, leading to 5 features to be selected for taining.
  If you choose a fixed integer, ensure all the feature sets under evaluation have atleast that many features.</p>
  </dd>
  </dl>
  </td></tr>
  <tr><td class="option-group" colspan="2">
  <kbd>-s, --sub_groups</kbd></td>
  </tr>
  <tr><td>&#160;</td><td><p class="first">This option allows the user to study different combinations of classes in a multi-class (N&gt;2) dataset.</p>
  <p>For example, in a dataset with 3 classes CN, FTD and AD,
  two studies of pair-wise combinations can be studied separately
  with the following flag <code class="docutils literal"><span class="pre">--sub_groups</span> <span class="pre">CN,FTD</span> <span class="pre">CN,AD</span></code>.
  This allows the user to focus on few interesting subgroups depending on their dataset/goal.</p>
  <p>Format: Different subgroups must be separated by space,
  and each sub-group must be a comma-separated list of class names defined in the meta data file.
  Hence it is strongly recommended to use class names without any spaces, commas, hyphens and special characters,
  and ideally just alphanumeric characters separated by underscores.</p>
  <p>Any number of subgroups can be specified, but each subgroup must have atleast two distinct classes.</p>
  <p class="last">Default: <code class="docutils literal"><span class="pre">'all'</span></code>, leading to inclusion of all available classes in a all-vs-all multi-class setting.</p>
  </td></tr>
  <tr><td class="option-group">
  <kbd>-g, --gs_level</kbd></td>
  <td><p class="first">Possible choices: none, light, exhaustive</p>
  <p>Flag to specify the level of grid search during hyper-parameter optimization on the training set.
  Allowed options are : ‘none’, ‘light’ and ‘exhaustive’, in the order of how many values/values will be optimized.</p>
  <p>More parameters and more values demand more resources and much longer time for optimization.</p>
  <dl class="last docutils">
  <dt>The ‘light’ option tries to “folk wisdom” to try least number of values (no more than one or two),</dt>
  <dd>for the parameters for the given classifier. (e.g. a lage number say 500 trees for a random forest optimization).
  The ‘light’ will be the fastest and should give a “rough idea” of predictive performance.
  The ‘exhaustive’ option will try to most parameter values for the most parameters that can be optimized.</dd>
  </dl>
  </td></tr>
  </tbody>
  </table>
  </div>
  <div class="section" id="Predictive Model">
  <h2>Predictive Model<a class="headerlink" href="#Predictive Model" title="Permalink to this headline">¶</a></h2>
  <p>Parameters related to pipeline comprising the predictive model</p>
  <table class="docutils option-list" frame="void" rules="none">
  <col class="option" />
  <col class="description" />
  <tbody valign="top">
  <tr><td class="option-group" colspan="2">
  <kbd>-fs, --feat_select_method</kbd></td>
  </tr>
  <tr><td>&#160;</td><td><p class="first">Possible choices: selectkbest_mutual_info_classif, selectkbest_f_classif, variancethreshold</p>
  <p>Feature selection method to apply prior to training the classifier.</p>
  <p class="last">Default: ‘VarianceThreshold’, removing features with 0.001 percent of lowest variance (zeros etc).</p>
  </td></tr>
  <tr><td class="option-group" colspan="2">
  <kbd>-e, --classifier</kbd></td>
  </tr>
  <tr><td>&#160;</td><td><p class="first">Possible choices: randomforestclassifier, extratreesclassifier, svm</p>
  <p>String specifying one of the implemented classifiers.
  (Classifiers are carefully chosen to allow for the comprehensive report provided by neuropredict).</p>
  <p class="last">Default: ‘RandomForestClassifier’</p>
  </td></tr>
  </tbody>
  </table>
  </div>
  <div class="section" id="Visualization">
  <h2>Visualization<a class="headerlink" href="#Visualization" title="Permalink to this headline">¶</a></h2>
  <p>Parameters related to generating visualizations</p>
  <table class="docutils option-list" frame="void" rules="none">
  <col class="option" />
  <col class="description" />
  <tbody valign="top">
  <tr><td class="option-group">
  <kbd>-z, --make_vis</kbd></td>
  <td>Option to make visualizations from existing results in the given path.
  This is helpful when neuropredict failed to generate result figures automatically
  e.g. on a HPC cluster, or another environment when DISPLAY is either not available.</td></tr>
  </tbody>
  </table>
  </div>
  <div class="section" id="Computing">
  <h2>Computing<a class="headerlink" href="#Computing" title="Permalink to this headline">¶</a></h2>
  <p>Parameters related to computations/debugging</p>
  <table class="docutils option-list" frame="void" rules="none">
  <col class="option" />
  <col class="description" />
  <tbody valign="top">
  <tr><td class="option-group" colspan="2">
  <kbd>-c, --num_procs</kbd></td>
  </tr>
  <tr><td>&#160;</td><td><p class="first">Number of CPUs to use to parallelize CV repetitions.</p>
  <p>Default : 4.</p>
  <p class="last">Number of CPUs will be capped at the number available on the machine if higher is requested.</p>
  </td></tr>
  <tr><td class="option-group">
  <kbd>-v, --version</kbd></td>
  <td>show program’s version number and exit</td></tr>
  </tbody>
  </table>
  </div>
