# Legal Notice for MELLM-DW-Suppression-Evaluation

## Copyright
This model is modified based on the open-source project Time-LLM. For the source code of the Time-LLM model, please refer to: https://github.com/KimMeen/Time-LLM

## Usage License
The MELLM (Multimodal Evaluation Large Language Model) code and related materials in this repository are provided under the [MIT License](LICENSE) (or other license: e.g., Apache 2.0, GPL-3.0). For detailed license terms, please refer to the LICENSE file in the root directory.

### Restrictions on Use
1. The MELLM model is intended for **academic research and non-commercial use only**. Commercial use (including but not limited to product development, profit-making services) requires prior written permission from the copyright holder.
2. Users must comply with relevant national laws and regulations when using the model, and shall not use the model for activities that violate public morality or legal provisions (e.g., harmful data prediction, unauthorized industrial application).

## Disclaimer
1. The MELLM model and related code are provided "as is" without any express or implied warranties, including but not limited to the warranties of merchantability, fitness for a particular purpose and non-infringement.
2. In no event shall the copyright holder be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including but not limited to procurement of substitute goods or services; loss of use, data, or profits; or business interruption) arising in any way out of the use of this software, even if advised of the possibility of such damage.
3. The prediction results of the MELLM model for DW explosion suppression performance are for reference only, and the copyright holder shall not be responsible for any losses caused by the use of the model results in actual engineering applications.

## Requirements
Use python 3.11 from MiniConda

torch==2.9.1
accelerate==0.28.0
einops==0.7.0
matplotlib==3.7.0
numpy==1.23.5
pandas==1.5.3
scikit_learn==1.2.2
scipy==1.12.0
tqdm==4.65.0
peft==0.4.0
transformers==4.31.0
sentencepiece==0.2.0

To install all dependencies:
```
pip install -r requirements.txt
```
If the versions of the aforementioned plugins are not applicable, you may install them according to your hardware requirements.

## Datasets
You can access the well pre-processed datasets from [[Google Drive]](https://drive.google.com/file/d/1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP/view?usp=sharing), then place the downloaded contents under `./dataset`

## Detailed usage

Please refer to ```run_main.py```, ```run_m4.py``` and ```run_pretrain.py``` for the detailed description of each hyperparameter.

## Contact
For legal questions regarding the use of this repository, please contact: [sh@lit.edu.cn]
