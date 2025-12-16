This is a Repository for Transformer robustness evaluation using Reinforcement Learning.

Please Ignore the LLM-Adapters, EzPC, and importance-aware-sparse-tuning-IST-paper in root directory. Sorry, but the code is DIRTY now!

## How to Start

### Allocate enough memory for Docker container
    mount -o remount,size=64G /dev/shm
### Activate python enviroment first
    conda activate llm_ist
### Go into the .sh located directory (/root/ppml/MoE-Privacy)
    cd /root/ppml/MoE-Privacy
### Execute the running scripts 
    bash llama_7B_LayerImportance.sh [lora_r] [lora_alpha] [logfile_path] [rl_lr] [degree]

lora_r: parameter for lora, ignore (we just use the Lora Framework to inference...), just set is to 32.  
lora_alpha: parameter for lora, ignore, just set is to 64.  
logfile_path: the log file output path, you can change it when the learning rate varies.  
rl_lr: reinforcement learning rate used in importance score update, now 20-40 is acceptable.  
degree: parameter for early debug, now deprecated. Just set it to 2.  

example: `bash llama_7B_LayerImportance.sh 32 
64 output.log 20 2`

#### Note: Though we call the script "llama_7B_LayerImportance.sh", we just evaluate the Bert-base model for different tasks now, please check out the .sh for more detials!

### The Result file 
The result outputs to file importance_scores_.....txt in /root/ppml/MoE-Privacy. You can modified the name in variable self.log_path in layer_importance_evaluator.py

### Stop the process
Cause running the sh now is using nohup, so we run it in backend.  
When you want interrupt it, run
`ps aux | grep rl_tune.py`
to check the process (rl_tune.py is the starting point of our evaluate, because we use the LLM-Adapter framework).  
And then kill the first process:
`kill -9 [process_id_of_rl_tune.py]` 

```
MoE-Privacy
├─ Bert-structure.txt
├─ EzPC
│  ├─ Aramis
│  │  ├─ 3party-aramis
│  │  │  ├─ party0
│  │  │  │  ├─ .config_HW_DEBUG_x64
│  │  │  │  ├─ App
│  │  │  │  │  ├─ App.cpp
│  │  │  │  │  ├─ App.h
│  │  │  │  │  ├─ Edger8rSyntax
│  │  │  │  │  │  ├─ Arrays.cpp
│  │  │  │  │  │  ├─ Functions.cpp
│  │  │  │  │  │  ├─ Pointers.cpp
│  │  │  │  │  │  └─ Types.cpp
│  │  │  │  │  ├─ IAS_report_verifier.cpp
│  │  │  │  │  ├─ IAS_report_verifier.h
│  │  │  │  │  ├─ TrustedLibrary
│  │  │  │  │  │  ├─ Libc.cpp
│  │  │  │  │  │  ├─ Libcxx.cpp
│  │  │  │  │  │  └─ Thread.cpp
│  │  │  │  │  ├─ app_sgx_instream.cpp
│  │  │  │  │  ├─ app_sgx_instream.h
│  │  │  │  │  ├─ app_sleep_utils.cpp
│  │  │  │  │  ├─ app_sleep_utils.h
│  │  │  │  │  ├─ attest_sequence.cpp
│  │  │  │  │  ├─ attest_sequence.h
│  │  │  │  │  ├─ basicSocketsPort.cpp
│  │  │  │  │  ├─ basicSocketsPort.h
│  │  │  │  │  ├─ benchmark_utils.h
│  │  │  │  │  ├─ cross_call_counter.cpp
│  │  │  │  │  ├─ cross_call_counter.h
│  │  │  │  │  ├─ input_and_key_handler.cpp
│  │  │  │  │  ├─ input_and_key_handler.h
│  │  │  │  │  ├─ mac_key_utils_app.cpp
│  │  │  │  │  ├─ mac_key_utils_app.h
│  │  │  │  │  ├─ print_prepends.cpp
│  │  │  │  │  ├─ print_prepends.h
│  │  │  │  │  ├─ quote_creator
│  │  │  │  │  │  ├─ quote_creator.cpp
│  │  │  │  │  │  └─ quote_creator.h
│  │  │  │  │  ├─ register_time.cpp
│  │  │  │  │  ├─ register_time.h
│  │  │  │  │  ├─ sgx_abort_handler.cpp
│  │  │  │  │  ├─ sgx_abort_handler.h
│  │  │  │  │  ├─ sgx_threads_ocall_handler.cpp
│  │  │  │  │  ├─ sgx_threads_ocall_handler.h
│  │  │  │  │  ├─ sgx_tprotected_fs.h
│  │  │  │  │  ├─ socket_interface.cpp
│  │  │  │  │  ├─ socket_interface.h
│  │  │  │  │  ├─ truce.cpp
│  │  │  │  │  ├─ truce_addresses.cpp
│  │  │  │  │  ├─ truce_addresses.h
│  │  │  │  │  ├─ truce_app.cpp
│  │  │  │  │  ├─ truce_app.h
│  │  │  │  │  ├─ truce_client.h
│  │  │  │  │  └─ truce_u.h
│  │  │  │  ├─ Enclave
│  │  │  │  │  ├─ .Enclave.cpp.swo
│  │  │  │  │  ├─ Edger8rSyntax
│  │  │  │  │  │  ├─ Arrays.cpp
│  │  │  │  │  │  ├─ Arrays.edl
│  │  │  │  │  │  ├─ Functions.cpp
│  │  │  │  │  │  ├─ Functions.edl
│  │  │  │  │  │  ├─ Pointers.cpp
│  │  │  │  │  │  ├─ Pointers.edl
│  │  │  │  │  │  ├─ Types.cpp
│  │  │  │  │  │  └─ Types.edl
│  │  │  │  │  ├─ Enclave.config.xml
│  │  │  │  │  ├─ Enclave.cpp
│  │  │  │  │  ├─ Enclave.edl
│  │  │  │  │  ├─ Enclave.h
│  │  │  │  │  ├─ Enclave.lds
│  │  │  │  │  ├─ Enclave_private.pem
│  │  │  │  │  ├─ IAS_report_verifier.cpp
│  │  │  │  │  ├─ IAS_report_verifier.h
│  │  │  │  │  ├─ TrustedLibrary
│  │  │  │  │  │  ├─ Libc.cpp
│  │  │  │  │  │  ├─ Libc.edl
│  │  │  │  │  │  ├─ Libcxx.cpp
│  │  │  │  │  │  ├─ Libcxx.edl
│  │  │  │  │  │  ├─ Thread.cpp
│  │  │  │  │  │  ├─ Thread.edl
│  │  │  │  │  │  ├─ sgx_tae_service.edl
│  │  │  │  │  │  ├─ sgx_tae_service.h
│  │  │  │  │  │  └─ sgx_tprotected_fs.edl
│  │  │  │  │  ├─ config.01.xml
│  │  │  │  │  ├─ config.02.xml
│  │  │  │  │  ├─ config.03.xml
│  │  │  │  │  ├─ config.04.xml
│  │  │  │  │  ├─ sgx_tprotected_fs.h
│  │  │  │  │  ├─ sigcounts.cpp
│  │  │  │  │  ├─ sigcounts.h
│  │  │  │  │  ├─ truce_enclave.cpp
│  │  │  │  │  ├─ truce_enclave.h
│  │  │  │  │  ├─ truce_private_keys.h
│  │  │  │  │  ├─ truce_t.h
│  │  │  │  │  └─ utils_ported_sgx.h
│  │  │  │  ├─ Include
│  │  │  │  │  └─ user_types.h
│  │  │  │  ├─ Makefile
│  │  │  │  ├─ aux_lib
│  │  │  │  │  ├─ Makefile
│  │  │  │  │  ├─ aux_funcs.cpp
│  │  │  │  │  ├─ aux_funcs.h
│  │  │  │  │  ├─ cpp-base64
│  │  │  │  │  │  ├─ LICENSE
│  │  │  │  │  │  ├─ README.md
│  │  │  │  │  │  ├─ base64.cpp
│  │  │  │  │  │  ├─ base64.h
│  │  │  │  │  │  ├─ compile-and-run-test
│  │  │  │  │  │  └─ test.cpp
│  │  │  │  │  ├─ cpp-base64.zip
│  │  │  │  │  └─ libaux_lib.a
│  │  │  │  ├─ compile_porthos_to_aramis.py
│  │  │  │  ├─ compile_to_aramis.sh
│  │  │  │  ├─ files
│  │  │  │  │  ├─ all_keys.cpp
│  │  │  │  │  ├─ all_keys.h
│  │  │  │  │  ├─ data
│  │  │  │  │  │  ├─ dataparsed.cpp
│  │  │  │  │  │  ├─ dataparsed.h
│  │  │  │  │  │  ├─ mnist_data_8_AC
│  │  │  │  │  │  ├─ mnist_data_8_BD
│  │  │  │  │  │  ├─ mnist_data_8_samples
│  │  │  │  │  │  ├─ mnist_labels_8_AC
│  │  │  │  │  │  ├─ mnist_labels_8_BD
│  │  │  │  │  │  ├─ mnist_labels_8_samples
│  │  │  │  │  │  └─ parsedata.py
│  │  │  │  │  ├─ keyA
│  │  │  │  │  ├─ keyAB
│  │  │  │  │  ├─ keyAC
│  │  │  │  │  ├─ keyAD
│  │  │  │  │  ├─ keyB
│  │  │  │  │  ├─ keyBC
│  │  │  │  │  ├─ keyBD
│  │  │  │  │  ├─ keyC
│  │  │  │  │  ├─ keyCD
│  │  │  │  │  └─ keyD
│  │  │  │  ├─ src
│  │  │  │  │  ├─ AESObject.cpp
│  │  │  │  │  ├─ AESObject.h
│  │  │  │  │  ├─ EzPCFunctionalities.cpp
│  │  │  │  │  ├─ EzPCFunctionalities.h
│  │  │  │  │  ├─ Functionalities.cpp
│  │  │  │  │  ├─ Functionalities.h
│  │  │  │  │  ├─ ParallelAESObject.cpp
│  │  │  │  │  ├─ ParallelAESObject.h
│  │  │  │  │  ├─ basicSockets.cpp
│  │  │  │  │  ├─ basicSockets.h
│  │  │  │  │  ├─ connect.cpp
│  │  │  │  │  ├─ connect.h
│  │  │  │  │  ├─ example_neural_nets
│  │  │  │  │  │  ├─ athos_securenn_nw_a.cpp
│  │  │  │  │  │  ├─ athos_securenn_nw_b.cpp
│  │  │  │  │  │  ├─ athos_securenn_nw_c.cpp
│  │  │  │  │  │  ├─ athos_securenn_nw_d.cpp
│  │  │  │  │  │  ├─ densenet_121.cpp
│  │  │  │  │  │  ├─ res_net_101.cpp
│  │  │  │  │  │  ├─ res_net_152.cpp
│  │  │  │  │  │  ├─ res_net_18.cpp
│  │  │  │  │  │  ├─ res_net_200.cpp
│  │  │  │  │  │  ├─ res_net_34.cpp
│  │  │  │  │  │  ├─ res_net_50.cpp
│  │  │  │  │  │  └─ squeezenet_image_net.cpp
│  │  │  │  │  ├─ ezpc.h
│  │  │  │  │  ├─ globals.h
│  │  │  │  │  ├─ main.cpp
│  │  │  │  │  ├─ main.h
│  │  │  │  │  ├─ network_config.h
│  │  │  │  │  ├─ optimizations_config.h
│  │  │  │  │  ├─ secondary.cpp
│  │  │  │  │  ├─ secondary.h
│  │  │  │  │  ├─ selective_run_config.h
│  │  │  │  │  ├─ tools.cpp
│  │  │  │  │  └─ tools.h
│  │  │  │  ├─ truce_headers
│  │  │  │  │  ├─ IAS_report.h
│  │  │  │  │  ├─ defs.h
│  │  │  │  │  ├─ truce_public_keys.h
│  │  │  │  │  └─ truce_record.h
│  │  │  │  ├─ update-all.sh
│  │  │  │  ├─ util
│  │  │  │  │  ├─ Config.h
│  │  │  │  │  ├─ TedKrovetzAesNiWrapperC.cpp
│  │  │  │  │  └─ TedKrovetzAesNiWrapperC.h
│  │  │  │  └─ utils_sgx_port
│  │  │  │     ├─ class_primary_keys.cpp
│  │  │  │     ├─ class_primary_keys.h
│  │  │  │     ├─ parse_circuitfile.py
│  │  │  │     ├─ parsed_circuitfile.cpp
│  │  │  │     ├─ parsed_circuitfile.h
│  │  │  │     ├─ port_utils_sgx.cpp
│  │  │  │     ├─ port_utils_sgx.h
│  │  │  │     ├─ sleep_utils_sgx.cpp
│  │  │  │     ├─ sleep_utils_sgx.h
│  │  │  │     ├─ utils_abort_sgx.cpp
│  │  │  │     ├─ utils_abort_sgx.h
│  │  │  │     ├─ utils_input_sgx.cpp
│  │  │  │     ├─ utils_input_sgx.h
│  │  │  │     ├─ utils_malloc_sgx.cpp
│  │  │  │     ├─ utils_malloc_sgx.h
│  │  │  │     ├─ utils_print_sgx.cpp
│  │  │  │     ├─ utils_print_sgx.h
│  │  │  │     ├─ utils_rand_sgx.cpp
│  │  │  │     ├─ utils_rand_sgx.h
│  │  │  │     ├─ utils_time_sgx.cpp
│  │  │  │     └─ utils_time_sgx.h
│  │  │  ├─ party1
│  │  │  │  ├─ .config_HW_DEBUG_x64
│  │  │  │  ├─ App
│  │  │  │  │  ├─ App.cpp
│  │  │  │  │  ├─ App.h
│  │  │  │  │  ├─ Edger8rSyntax
│  │  │  │  │  │  ├─ Arrays.cpp
│  │  │  │  │  │  ├─ Functions.cpp
│  │  │  │  │  │  ├─ Pointers.cpp
│  │  │  │  │  │  └─ Types.cpp
│  │  │  │  │  ├─ IAS_report_verifier.cpp
│  │  │  │  │  ├─ IAS_report_verifier.h
│  │  │  │  │  ├─ TrustedLibrary
│  │  │  │  │  │  ├─ Libc.cpp
│  │  │  │  │  │  ├─ Libcxx.cpp
│  │  │  │  │  │  └─ Thread.cpp
│  │  │  │  │  ├─ app_sgx_instream.cpp
│  │  │  │  │  ├─ app_sgx_instream.h
│  │  │  │  │  ├─ app_sleep_utils.cpp
│  │  │  │  │  ├─ app_sleep_utils.h
│  │  │  │  │  ├─ attest_sequence.cpp
│  │  │  │  │  ├─ attest_sequence.h
│  │  │  │  │  ├─ basicSocketsPort.cpp
│  │  │  │  │  ├─ basicSocketsPort.h
│  │  │  │  │  ├─ benchmark_utils.h
│  │  │  │  │  ├─ cross_call_counter.cpp
│  │  │  │  │  ├─ cross_call_counter.h
│  │  │  │  │  ├─ input_and_key_handler.cpp
│  │  │  │  │  ├─ input_and_key_handler.h
│  │  │  │  │  ├─ mac_key_utils_app.cpp
│  │  │  │  │  ├─ mac_key_utils_app.h
│  │  │  │  │  ├─ print_prepends.cpp
│  │  │  │  │  ├─ print_prepends.h
│  │  │  │  │  ├─ quote_creator
│  │  │  │  │  │  ├─ quote_creator.cpp
│  │  │  │  │  │  └─ quote_creator.h
│  │  │  │  │  ├─ register_time.cpp
│  │  │  │  │  ├─ register_time.h
│  │  │  │  │  ├─ sgx_abort_handler.cpp
│  │  │  │  │  ├─ sgx_abort_handler.h
│  │  │  │  │  ├─ sgx_threads_ocall_handler.cpp
│  │  │  │  │  ├─ sgx_threads_ocall_handler.h
│  │  │  │  │  ├─ sgx_tprotected_fs.h
│  │  │  │  │  ├─ socket_interface.cpp
│  │  │  │  │  ├─ socket_interface.h
│  │  │  │  │  ├─ truce.cpp
│  │  │  │  │  ├─ truce_addresses.cpp
│  │  │  │  │  ├─ truce_addresses.h
│  │  │  │  │  ├─ truce_app.cpp
│  │  │  │  │  ├─ truce_app.h
│  │  │  │  │  ├─ truce_client.h
│  │  │  │  │  └─ truce_u.h
│  │  │  │  ├─ Enclave
│  │  │  │  │  ├─ .Enclave.cpp.swo
│  │  │  │  │  ├─ Edger8rSyntax
│  │  │  │  │  │  ├─ Arrays.cpp
│  │  │  │  │  │  ├─ Arrays.edl
│  │  │  │  │  │  ├─ Functions.cpp
│  │  │  │  │  │  ├─ Functions.edl
│  │  │  │  │  │  ├─ Pointers.cpp
│  │  │  │  │  │  ├─ Pointers.edl
│  │  │  │  │  │  ├─ Types.cpp
│  │  │  │  │  │  └─ Types.edl
│  │  │  │  │  ├─ Enclave.config.xml
│  │  │  │  │  ├─ Enclave.cpp
│  │  │  │  │  ├─ Enclave.edl
│  │  │  │  │  ├─ Enclave.h
│  │  │  │  │  ├─ Enclave.lds
│  │  │  │  │  ├─ Enclave_private.pem
│  │  │  │  │  ├─ IAS_report_verifier.cpp
│  │  │  │  │  ├─ IAS_report_verifier.h
│  │  │  │  │  ├─ TrustedLibrary
│  │  │  │  │  │  ├─ Libc.cpp
│  │  │  │  │  │  ├─ Libc.edl
│  │  │  │  │  │  ├─ Libcxx.cpp
│  │  │  │  │  │  ├─ Libcxx.edl
│  │  │  │  │  │  ├─ Thread.cpp
│  │  │  │  │  │  ├─ Thread.edl
│  │  │  │  │  │  ├─ sgx_tae_service.edl
│  │  │  │  │  │  ├─ sgx_tae_service.h
│  │  │  │  │  │  └─ sgx_tprotected_fs.edl
│  │  │  │  │  ├─ config.01.xml
│  │  │  │  │  ├─ config.02.xml
│  │  │  │  │  ├─ config.03.xml
│  │  │  │  │  ├─ config.04.xml
│  │  │  │  │  ├─ sgx_tprotected_fs.h
│  │  │  │  │  ├─ sigcounts.cpp
│  │  │  │  │  ├─ sigcounts.h
│  │  │  │  │  ├─ truce_enclave.cpp
│  │  │  │  │  ├─ truce_enclave.h
│  │  │  │  │  ├─ truce_private_keys.h
│  │  │  │  │  ├─ truce_t.h
│  │  │  │  │  └─ utils_ported_sgx.h
│  │  │  │  ├─ Include
│  │  │  │  │  └─ user_types.h
│  │  │  │  ├─ Makefile
│  │  │  │  ├─ aux_lib
│  │  │  │  │  ├─ Makefile
│  │  │  │  │  ├─ aux_funcs.cpp
│  │  │  │  │  ├─ aux_funcs.h
│  │  │  │  │  ├─ cpp-base64
│  │  │  │  │  │  ├─ LICENSE
│  │  │  │  │  │  ├─ README.md
│  │  │  │  │  │  ├─ base64.cpp
│  │  │  │  │  │  ├─ base64.h
│  │  │  │  │  │  ├─ compile-and-run-test
│  │  │  │  │  │  └─ test.cpp
│  │  │  │  │  ├─ cpp-base64.zip
│  │  │  │  │  └─ libaux_lib.a
│  │  │  │  ├─ compile_porthos_to_aramis.py
│  │  │  │  ├─ files
│  │  │  │  │  ├─ all_keys.cpp
│  │  │  │  │  ├─ all_keys.h
│  │  │  │  │  ├─ data
│  │  │  │  │  │  ├─ dataparsed.cpp
│  │  │  │  │  │  ├─ dataparsed.h
│  │  │  │  │  │  ├─ mnist_data_8_AC
│  │  │  │  │  │  ├─ mnist_data_8_BD
│  │  │  │  │  │  ├─ mnist_data_8_samples
│  │  │  │  │  │  ├─ mnist_labels_8_AC
│  │  │  │  │  │  ├─ mnist_labels_8_BD
│  │  │  │  │  │  ├─ mnist_labels_8_samples
│  │  │  │  │  │  └─ parsedata.py
│  │  │  │  │  ├─ keyA
│  │  │  │  │  ├─ keyAB
│  │  │  │  │  ├─ keyAC
│  │  │  │  │  ├─ keyAD
│  │  │  │  │  ├─ keyB
│  │  │  │  │  ├─ keyBC
│  │  │  │  │  ├─ keyBD
│  │  │  │  │  ├─ keyC
│  │  │  │  │  ├─ keyCD
│  │  │  │  │  └─ keyD
│  │  │  │  ├─ src
│  │  │  │  │  ├─ AESObject.cpp
│  │  │  │  │  ├─ AESObject.h
│  │  │  │  │  ├─ EzPCFunctionalities.cpp
│  │  │  │  │  ├─ EzPCFunctionalities.h
│  │  │  │  │  ├─ Functionalities.cpp
│  │  │  │  │  ├─ Functionalities.h
│  │  │  │  │  ├─ ParallelAESObject.cpp
│  │  │  │  │  ├─ ParallelAESObject.h
│  │  │  │  │  ├─ basicSockets.cpp
│  │  │  │  │  ├─ basicSockets.h
│  │  │  │  │  ├─ connect.cpp
│  │  │  │  │  ├─ connect.h
│  │  │  │  │  ├─ example_neural_nets
│  │  │  │  │  │  ├─ athos_securenn_nw_a.cpp
│  │  │  │  │  │  ├─ athos_securenn_nw_b.cpp
│  │  │  │  │  │  ├─ athos_securenn_nw_c.cpp
│  │  │  │  │  │  ├─ athos_securenn_nw_d.cpp
│  │  │  │  │  │  ├─ densenet_121.cpp
│  │  │  │  │  │  ├─ res_net_101.cpp
│  │  │  │  │  │  ├─ res_net_152.cpp
│  │  │  │  │  │  ├─ res_net_18.cpp
│  │  │  │  │  │  ├─ res_net_200.cpp
│  │  │  │  │  │  ├─ res_net_34.cpp
│  │  │  │  │  │  ├─ res_net_50.cpp
│  │  │  │  │  │  └─ squeezenet_image_net.cpp
│  │  │  │  │  ├─ ezpc.h
│  │  │  │  │  ├─ globals.h
│  │  │  │  │  ├─ main.cpp
│  │  │  │  │  ├─ main.h
│  │  │  │  │  ├─ network_config.h
│  │  │  │  │  ├─ optimizations_config.h
│  │  │  │  │  ├─ secondary.cpp
│  │  │  │  │  ├─ secondary.h
│  │  │  │  │  ├─ selective_run_config.h
│  │  │  │  │  ├─ tools.cpp
│  │  │  │  │  └─ tools.h
│  │  │  │  ├─ truce_headers
│  │  │  │  │  ├─ IAS_report.h
│  │  │  │  │  ├─ defs.h
│  │  │  │  │  ├─ truce_public_keys.h
│  │  │  │  │  └─ truce_record.h
│  │  │  │  ├─ util
│  │  │  │  │  ├─ Config.h
│  │  │  │  │  ├─ TedKrovetzAesNiWrapperC.cpp
│  │  │  │  │  └─ TedKrovetzAesNiWrapperC.h
│  │  │  │  └─ utils_sgx_port
│  │  │  │     ├─ class_primary_keys.cpp
│  │  │  │     ├─ class_primary_keys.h
│  │  │  │     ├─ parse_circuitfile.py
│  │  │  │     ├─ parsed_circuitfile.cpp
│  │  │  │     ├─ parsed_circuitfile.h
│  │  │  │     ├─ port_utils_sgx.cpp
│  │  │  │     ├─ port_utils_sgx.h
│  │  │  │     ├─ sleep_utils_sgx.cpp
│  │  │  │     ├─ sleep_utils_sgx.h
│  │  │  │     ├─ utils_abort_sgx.cpp
│  │  │  │     ├─ utils_abort_sgx.h
│  │  │  │     ├─ utils_input_sgx.cpp
│  │  │  │     ├─ utils_input_sgx.h
│  │  │  │     ├─ utils_malloc_sgx.cpp
│  │  │  │     ├─ utils_malloc_sgx.h
│  │  │  │     ├─ utils_print_sgx.cpp
│  │  │  │     ├─ utils_print_sgx.h
│  │  │  │     ├─ utils_rand_sgx.cpp
│  │  │  │     ├─ utils_rand_sgx.h
│  │  │  │     ├─ utils_time_sgx.cpp
│  │  │  │     └─ utils_time_sgx.h
│  │  │  ├─ party2
│  │  │  │  ├─ .config_HW_DEBUG_x64
│  │  │  │  ├─ App
│  │  │  │  │  ├─ App.cpp
│  │  │  │  │  ├─ App.h
│  │  │  │  │  ├─ Edger8rSyntax
│  │  │  │  │  │  ├─ Arrays.cpp
│  │  │  │  │  │  ├─ Functions.cpp
│  │  │  │  │  │  ├─ Pointers.cpp
│  │  │  │  │  │  └─ Types.cpp
│  │  │  │  │  ├─ IAS_report_verifier.cpp
│  │  │  │  │  ├─ IAS_report_verifier.h
│  │  │  │  │  ├─ TrustedLibrary
│  │  │  │  │  │  ├─ Libc.cpp
│  │  │  │  │  │  ├─ Libcxx.cpp
│  │  │  │  │  │  └─ Thread.cpp
│  │  │  │  │  ├─ app_sgx_instream.cpp
│  │  │  │  │  ├─ app_sgx_instream.h
│  │  │  │  │  ├─ app_sleep_utils.cpp
│  │  │  │  │  ├─ app_sleep_utils.h
│  │  │  │  │  ├─ attest_sequence.cpp
│  │  │  │  │  ├─ attest_sequence.h
│  │  │  │  │  ├─ basicSocketsPort.cpp
│  │  │  │  │  ├─ basicSocketsPort.h
│  │  │  │  │  ├─ benchmark_utils.h
│  │  │  │  │  ├─ cross_call_counter.cpp
│  │  │  │  │  ├─ cross_call_counter.h
│  │  │  │  │  ├─ input_and_key_handler.cpp
│  │  │  │  │  ├─ input_and_key_handler.h
│  │  │  │  │  ├─ mac_key_utils_app.cpp
│  │  │  │  │  ├─ mac_key_utils_app.h
│  │  │  │  │  ├─ print_prepends.cpp
│  │  │  │  │  ├─ print_prepends.h
│  │  │  │  │  ├─ quote_creator
│  │  │  │  │  │  ├─ quote_creator.cpp
│  │  │  │  │  │  └─ quote_creator.h
│  │  │  │  │  ├─ register_time.cpp
│  │  │  │  │  ├─ register_time.h
│  │  │  │  │  ├─ sgx_abort_handler.cpp
│  │  │  │  │  ├─ sgx_abort_handler.h
│  │  │  │  │  ├─ sgx_threads_ocall_handler.cpp
│  │  │  │  │  ├─ sgx_threads_ocall_handler.h
│  │  │  │  │  ├─ sgx_tprotected_fs.h
│  │  │  │  │  ├─ socket_interface.cpp
│  │  │  │  │  ├─ socket_interface.h
│  │  │  │  │  ├─ truce.cpp
│  │  │  │  │  ├─ truce_addresses.cpp
│  │  │  │  │  ├─ truce_addresses.h
│  │  │  │  │  ├─ truce_app.cpp
│  │  │  │  │  ├─ truce_app.h
│  │  │  │  │  ├─ truce_client.h
│  │  │  │  │  └─ truce_u.h
│  │  │  │  ├─ Enclave
│  │  │  │  │  ├─ .Enclave.cpp.swo
│  │  │  │  │  ├─ Edger8rSyntax
│  │  │  │  │  │  ├─ Arrays.cpp
│  │  │  │  │  │  ├─ Arrays.edl
│  │  │  │  │  │  ├─ Functions.cpp
│  │  │  │  │  │  ├─ Functions.edl
│  │  │  │  │  │  ├─ Pointers.cpp
│  │  │  │  │  │  ├─ Pointers.edl
│  │  │  │  │  │  ├─ Types.cpp
│  │  │  │  │  │  └─ Types.edl
│  │  │  │  │  ├─ Enclave.config.xml
│  │  │  │  │  ├─ Enclave.cpp
│  │  │  │  │  ├─ Enclave.edl
│  │  │  │  │  ├─ Enclave.h
│  │  │  │  │  ├─ Enclave.lds
│  │  │  │  │  ├─ Enclave_private.pem
│  │  │  │  │  ├─ IAS_report_verifier.cpp
│  │  │  │  │  ├─ IAS_report_verifier.h
│  │  │  │  │  ├─ TrustedLibrary
│  │  │  │  │  │  ├─ Libc.cpp
│  │  │  │  │  │  ├─ Libc.edl
│  │  │  │  │  │  ├─ Libcxx.cpp
│  │  │  │  │  │  ├─ Libcxx.edl
│  │  │  │  │  │  ├─ Thread.cpp
│  │  │  │  │  │  ├─ Thread.edl
│  │  │  │  │  │  ├─ sgx_tae_service.edl
│  │  │  │  │  │  ├─ sgx_tae_service.h
│  │  │  │  │  │  └─ sgx_tprotected_fs.edl
│  │  │  │  │  ├─ config.01.xml
│  │  │  │  │  ├─ config.02.xml
│  │  │  │  │  ├─ config.03.xml
│  │  │  │  │  ├─ config.04.xml
│  │  │  │  │  ├─ sgx_tprotected_fs.h
│  │  │  │  │  ├─ sigcounts.cpp
│  │  │  │  │  ├─ sigcounts.h
│  │  │  │  │  ├─ truce_enclave.cpp
│  │  │  │  │  ├─ truce_enclave.h
│  │  │  │  │  ├─ truce_private_keys.h
│  │  │  │  │  ├─ truce_t.h
│  │  │  │  │  └─ utils_ported_sgx.h
│  │  │  │  ├─ Include
│  │  │  │  │  └─ user_types.h
│  │  │  │  ├─ Makefile
│  │  │  │  ├─ aux_lib
│  │  │  │  │  ├─ Makefile
│  │  │  │  │  ├─ aux_funcs.cpp
│  │  │  │  │  ├─ aux_funcs.h
│  │  │  │  │  ├─ cpp-base64
│  │  │  │  │  │  ├─ LICENSE
│  │  │  │  │  │  ├─ README.md
│  │  │  │  │  │  ├─ base64.cpp
│  │  │  │  │  │  ├─ base64.h
│  │  │  │  │  │  ├─ compile-and-run-test
│  │  │  │  │  │  └─ test.cpp
│  │  │  │  │  ├─ cpp-base64.zip
│  │  │  │  │  └─ libaux_lib.a
│  │  │  │  ├─ compile_porthos_to_aramis.py
│  │  │  │  ├─ files
│  │  │  │  │  ├─ all_keys.cpp
│  │  │  │  │  ├─ all_keys.h
│  │  │  │  │  ├─ data
│  │  │  │  │  │  ├─ dataparsed.cpp
│  │  │  │  │  │  ├─ dataparsed.h
│  │  │  │  │  │  ├─ mnist_data_8_AC
│  │  │  │  │  │  ├─ mnist_data_8_BD
│  │  │  │  │  │  ├─ mnist_data_8_samples
│  │  │  │  │  │  ├─ mnist_labels_8_AC
│  │  │  │  │  │  ├─ mnist_labels_8_BD
│  │  │  │  │  │  ├─ mnist_labels_8_samples
│  │  │  │  │  │  └─ parsedata.py
│  │  │  │  │  ├─ keyA
│  │  │  │  │  ├─ keyAB
│  │  │  │  │  ├─ keyAC
│  │  │  │  │  ├─ keyAD
│  │  │  │  │  ├─ keyB
│  │  │  │  │  ├─ keyBC
│  │  │  │  │  ├─ keyBD
│  │  │  │  │  ├─ keyC
│  │  │  │  │  ├─ keyCD
│  │  │  │  │  └─ keyD
│  │  │  │  ├─ src
│  │  │  │  │  ├─ AESObject.cpp
│  │  │  │  │  ├─ AESObject.h
│  │  │  │  │  ├─ EzPCFunctionalities.cpp
│  │  │  │  │  ├─ EzPCFunctionalities.h
│  │  │  │  │  ├─ Functionalities.cpp
│  │  │  │  │  ├─ Functionalities.h
│  │  │  │  │  ├─ ParallelAESObject.cpp
│  │  │  │  │  ├─ ParallelAESObject.h
│  │  │  │  │  ├─ basicSockets.cpp
│  │  │  │  │  ├─ basicSockets.h
│  │  │  │  │  ├─ connect.cpp
│  │  │  │  │  ├─ connect.h
│  │  │  │  │  ├─ example_neural_nets
│  │  │  │  │  │  ├─ athos_securenn_nw_a.cpp
│  │  │  │  │  │  ├─ athos_securenn_nw_b.cpp
│  │  │  │  │  │  ├─ athos_securenn_nw_c.cpp
│  │  │  │  │  │  ├─ athos_securenn_nw_d.cpp
│  │  │  │  │  │  ├─ densenet_121.cpp
│  │  │  │  │  │  ├─ res_net_101.cpp
│  │  │  │  │  │  ├─ res_net_152.cpp
│  │  │  │  │  │  ├─ res_net_18.cpp
│  │  │  │  │  │  ├─ res_net_200.cpp
│  │  │  │  │  │  ├─ res_net_34.cpp
│  │  │  │  │  │  ├─ res_net_50.cpp
│  │  │  │  │  │  └─ squeezenet_image_net.cpp
│  │  │  │  │  ├─ ezpc.h
│  │  │  │  │  ├─ globals.h
│  │  │  │  │  ├─ main.cpp
│  │  │  │  │  ├─ main.h
│  │  │  │  │  ├─ network_config.h
│  │  │  │  │  ├─ optimizations_config.h
│  │  │  │  │  ├─ secondary.cpp
│  │  │  │  │  ├─ secondary.h
│  │  │  │  │  ├─ selective_run_config.h
│  │  │  │  │  ├─ tools.cpp
│  │  │  │  │  └─ tools.h
│  │  │  │  ├─ truce_headers
│  │  │  │  │  ├─ IAS_report.h
│  │  │  │  │  ├─ defs.h
│  │  │  │  │  ├─ truce_public_keys.h
│  │  │  │  │  └─ truce_record.h
│  │  │  │  ├─ util
│  │  │  │  │  ├─ Config.h
│  │  │  │  │  ├─ TedKrovetzAesNiWrapperC.cpp
│  │  │  │  │  └─ TedKrovetzAesNiWrapperC.h
│  │  │  │  └─ utils_sgx_port
│  │  │  │     ├─ class_primary_keys.cpp
│  │  │  │     ├─ class_primary_keys.h
│  │  │  │     ├─ parse_circuitfile.py
│  │  │  │     ├─ parsed_circuitfile.cpp
│  │  │  │     ├─ parsed_circuitfile.h
│  │  │  │     ├─ port_utils_sgx.cpp
│  │  │  │     ├─ port_utils_sgx.h
│  │  │  │     ├─ sleep_utils_sgx.cpp
│  │  │  │     ├─ sleep_utils_sgx.h
│  │  │  │     ├─ utils_abort_sgx.cpp
│  │  │  │     ├─ utils_abort_sgx.h
│  │  │  │     ├─ utils_input_sgx.cpp
│  │  │  │     ├─ utils_input_sgx.h
│  │  │  │     ├─ utils_malloc_sgx.cpp
│  │  │  │     ├─ utils_malloc_sgx.h
│  │  │  │     ├─ utils_print_sgx.cpp
│  │  │  │     ├─ utils_print_sgx.h
│  │  │  │     ├─ utils_rand_sgx.cpp
│  │  │  │     ├─ utils_rand_sgx.h
│  │  │  │     ├─ utils_time_sgx.cpp
│  │  │  │     └─ utils_time_sgx.h
│  │  │  └─ service-provider
│  │  │     ├─ IAS_report.h
│  │  │     ├─ IAS_web_service
│  │  │     │  ├─ IAS_web_service.cpp
│  │  │     │  ├─ IAS_web_service.h
│  │  │     │  └─ cert_and_key.pem
│  │  │     ├─ Makefile
│  │  │     ├─ aux_lib
│  │  │     │  ├─ Makefile
│  │  │     │  ├─ aux_funcs.cpp
│  │  │     │  ├─ aux_funcs.h
│  │  │     │  ├─ cpp-base64
│  │  │     │  │  ├─ LICENSE
│  │  │     │  │  ├─ README.md
│  │  │     │  │  ├─ base64.cpp
│  │  │     │  │  ├─ base64.h
│  │  │     │  │  ├─ compile-and-run-test
│  │  │     │  │  └─ test.cpp
│  │  │     │  ├─ cpp-base64.zip
│  │  │     │  └─ libaux_lib.a
│  │  │     ├─ cert_and_key.pem
│  │  │     ├─ defs.h
│  │  │     ├─ server.cpp
│  │  │     ├─ truce_public_keys.h
│  │  │     ├─ truce_record.h
│  │  │     └─ truce_server
│  │  └─ README.md
│  ├─ Athos
│  │  ├─ CompileONNXGraph.py
│  │  ├─ CompileRandomForests.py
│  │  ├─ CompileSampleNetworks.py
│  │  ├─ CompileTF.sh
│  │  ├─ CompileTFGraph.py
│  │  ├─ CompilerScripts
│  │  │  ├─ change_onnx_output.py
│  │  │  ├─ comparison_scripts
│  │  │  │  ├─ compare_np_arrs.py
│  │  │  │  ├─ compare_output.py
│  │  │  │  ├─ compare_output.sh
│  │  │  │  ├─ convert_scale.py
│  │  │  │  ├─ convert_to_signed.py
│  │  │  │  └─ convert_to_signed.sh
│  │  │  ├─ compile_tf.py
│  │  │  ├─ compile_tf_graph.py
│  │  │  ├─ convert_keras_to_onnx.py
│  │  │  ├─ convert_keras_to_tf.py
│  │  │  ├─ convert_np_to_fixedpt.py
│  │  │  ├─ convert_saved_model_to_frozen_graph.py
│  │  │  ├─ create_tf_input.py
│  │  │  ├─ generate_concat.py
│  │  │  ├─ get_output.py
│  │  │  ├─ get_pred_tf_graph.py
│  │  │  ├─ grappler.py
│  │  │  ├─ memory_estimate.py
│  │  │  ├─ onnx_replace_subgraph.py
│  │  │  ├─ parse_config.py
│  │  │  ├─ preprocess_frozen_tf_graph.py
│  │  │  ├─ remove_tf_nodes.py
│  │  │  ├─ replace_tf_nodes_with_identity.py
│  │  │  ├─ sample_networks
│  │  │  │  ├─ print_stats_2pc.sh
│  │  │  │  ├─ print_stats_3pc.sh
│  │  │  │  ├─ print_stats_cpp.sh
│  │  │  │  ├─ run_demo_2pc.sh
│  │  │  │  ├─ run_demo_3pc.sh
│  │  │  │  └─ run_demo_cpp.sh
│  │  │  ├─ tf_graph_io.py
│  │  │  └─ tf_graph_trans.py
│  │  ├─ HelperScripts
│  │  │  ├─ CIFAR10
│  │  │  ├─ CheXpert
│  │  │  ├─ Confirm_preprocessing.py
│  │  │  ├─ Convert_WnId_To_TrainId.py
│  │  │  ├─ FindAccuracy.py
│  │  │  ├─ FindAccuracy_Porthos.py
│  │  │  ├─ FindAccuracy_TF.py
│  │  │  ├─ ImageNet_ValData
│  │  │  ├─ Prepare_ImageNet_Val.sh
│  │  │  ├─ Random_Image_Selection.py
│  │  │  ├─ RunPorthosInference_ImageNet.sh
│  │  │  ├─ Scale_img_and_model.py
│  │  │  ├─ SetupCIFAR10.sh
│  │  │  ├─ SetupCheXpert.sh
│  │  │  ├─ nn_maxmintest.py
│  │  │  └─ pre_commit_format_python.sh
│  │  ├─ Networks
│  │  │  ├─ CheXpert
│  │  │  │  ├─ Data_batch
│  │  │  │  ├─ PreProcessedImages
│  │  │  │  ├─ Prepare_Model.py
│  │  │  │  ├─ Util.py
│  │  │  │  └─ setup_and_run.sh
│  │  │  ├─ ChestXRay
│  │  │  │  ├─ ChestXRay_tf_main.py
│  │  │  │  ├─ PreTrainedModel
│  │  │  │  │  ├─ KerasModel
│  │  │  │  │  └─ TFModel
│  │  │  │  ├─ README.md
│  │  │  │  ├─ SampleImages
│  │  │  │  │  └─ 00014251_029.png
│  │  │  │  ├─ SetupPretrainedTFModel.sh
│  │  │  │  └─ setup_and_run.sh
│  │  │  ├─ DenseNet
│  │  │  │  ├─ AccuracyAnalysisHelper
│  │  │  │  │  ├─ DenseNet64_acc_test.cpp
│  │  │  │  │  ├─ DenseNet_main_float_acc.py
│  │  │  │  │  ├─ InferenceErrors
│  │  │  │  │  ├─ InferenceOutputs
│  │  │  │  │  ├─ Paths.config
│  │  │  │  │  ├─ PreProcess_ImageNet.sh
│  │  │  │  │  ├─ PreProcessedImages
│  │  │  │  │  ├─ RunInference_ImageNet.sh
│  │  │  │  │  └─ helper_temp.sh
│  │  │  │  ├─ DenseNet_main.py
│  │  │  │  ├─ DenseNet_main_64_cpp_maxmintest.cpp
│  │  │  │  ├─ PreProcessingImages
│  │  │  │  │  ├─ DenseNet_preprocess_main.py
│  │  │  │  │  └─ DenseNet_preprocessing.py
│  │  │  │  ├─ PreTrainedModel
│  │  │  │  ├─ README.md
│  │  │  │  ├─ SampleImages
│  │  │  │  │  └─ n02109961_36_denseNet_preprocessed.pkl
│  │  │  │  ├─ densenet.py
│  │  │  │  ├─ nets_factory.py
│  │  │  │  └─ setup_and_run.sh
│  │  │  ├─ Lenet
│  │  │  │  ├─ README.md
│  │  │  │  ├─ TrainedModel
│  │  │  │  ├─ lenetLarge_mnist_inference.py
│  │  │  │  ├─ lenetLarge_mnist_train.py
│  │  │  │  ├─ lenetSmall_mnist_inference.py
│  │  │  │  └─ lenetSmall_mnist_train.py
│  │  │  ├─ LogisticRegression
│  │  │  │  ├─ LogisticRegressionInfer.py
│  │  │  │  ├─ LogisticRegressionTrain.py
│  │  │  │  ├─ README.md
│  │  │  │  └─ TrainedModel
│  │  │  ├─ OtherBenchmarks
│  │  │  │  ├─ MiniONN_CIFAR.py
│  │  │  │  ├─ README.md
│  │  │  │  └─ resnet32_cifar100.py
│  │  │  ├─ ResNet
│  │  │  │  ├─ AccuracyAnalysisHelper
│  │  │  │  │  ├─ InferenceErrors
│  │  │  │  │  ├─ InferenceOutputs
│  │  │  │  │  ├─ Paths.config
│  │  │  │  │  ├─ PreProcess_ImageNet.sh
│  │  │  │  │  ├─ PreProcessedImages
│  │  │  │  │  ├─ ResNet_main_float_acc.py
│  │  │  │  │  ├─ Resnet64_acc_test.cpp
│  │  │  │  │  └─ RunInference_ImageNet.sh
│  │  │  │  ├─ PreProcessingImages
│  │  │  │  │  ├─ ResNet_preprocess_main.py
│  │  │  │  │  └─ imagenet_preprocessing.py
│  │  │  │  ├─ PreTrainedModel
│  │  │  │  ├─ README.md
│  │  │  │  ├─ ResNet_main.py
│  │  │  │  ├─ ResNet_main_64_cpp_maxmintest.cpp
│  │  │  │  ├─ Resnet_Model.py
│  │  │  │  ├─ SampleImages
│  │  │  │  │  ├─ n02109961_36.JPEG
│  │  │  │  │  ├─ n02109961_36.xml
│  │  │  │  │  └─ n02109961_36_enc.pkl
│  │  │  │  └─ setup_and_run.sh
│  │  │  ├─ SecureNNBenchmarks
│  │  │  │  ├─ NetworkA.py
│  │  │  │  ├─ NetworkB.py
│  │  │  │  ├─ NetworkC.py
│  │  │  │  ├─ NetworkD.py
│  │  │  │  └─ README.md
│  │  │  ├─ SqueezeNetCIFAR10
│  │  │  │  ├─ PreProcessedImages
│  │  │  │  ├─ README.md
│  │  │  │  ├─ Squeezenet_model.py
│  │  │  │  ├─ TrainedModel
│  │  │  │  ├─ Util.py
│  │  │  │  └─ setup_and_run.sh
│  │  │  ├─ SqueezeNetImgNet
│  │  │  │  ├─ AccuracyAnalysisHelper
│  │  │  │  │  ├─ InferenceErrors
│  │  │  │  │  ├─ InferenceOutputs
│  │  │  │  │  ├─ Paths.config
│  │  │  │  │  ├─ PreProcess_ImageNet.sh
│  │  │  │  │  ├─ PreProcessedImages
│  │  │  │  │  ├─ RunInference_ImageNet.sh
│  │  │  │  │  ├─ SqueezeNet_main_float_acc.py
│  │  │  │  │  └─ squeezenet64_acc_test.cpp
│  │  │  │  ├─ PreProcessingImages
│  │  │  │  │  └─ SqNetImgNet_preprocess_main.py
│  │  │  │  ├─ PreTrainedModel
│  │  │  │  ├─ README.md
│  │  │  │  ├─ SampleImages
│  │  │  │  │  └─ n02109961_36.JPEG
│  │  │  │  ├─ setup_and_run.sh
│  │  │  │  ├─ squeezenet_main.py
│  │  │  │  ├─ squeezenet_main_64_cpp_maxmintest.cpp
│  │  │  │  └─ synset_words.txt
│  │  │  └─ sample_network.config
│  │  ├─ ONNXCompiler
│  │  │  ├─ ONNXNodesAST.py
│  │  │  ├─ Readme.md
│  │  │  ├─ __init__.py
│  │  │  ├─ common.py
│  │  │  ├─ config.json
│  │  │  ├─ create_input.py
│  │  │  ├─ onnx_run.py
│  │  │  ├─ onnx_run_tf.py
│  │  │  ├─ process_onnx.py
│  │  │  └─ test
│  │  │     ├─ __init__.py
│  │  │     └─ test.py
│  │  ├─ Paths.config
│  │  ├─ README.md
│  │  ├─ RandomForests
│  │  │  ├─ README.md
│  │  │  ├─ convert_pickle_to_graphviz.py
│  │  │  ├─ notebooks
│  │  │  │  ├─ RandomForestCaliforniaHousingPickle.ipynb
│  │  │  │  ├─ RandomForestCaliforniaHousingPickle.py
│  │  │  │  └─ housing.csv
│  │  │  ├─ parse_graphviz_to_ezpc_input.py
│  │  │  ├─ patch_ezpc_code_params.py
│  │  │  └─ random_forest_base_file.ezpc
│  │  ├─ SeeDot
│  │  │  ├─ AST
│  │  │  │  ├─ AST.py
│  │  │  │  ├─ ASTVisitor.py
│  │  │  │  ├─ IRBuilderAST.py
│  │  │  │  ├─ MtdAST.py
│  │  │  │  └─ PrintAST.py
│  │  │  ├─ Codegen
│  │  │  │  ├─ CodegenBase.py
│  │  │  │  └─ EzPC.py
│  │  │  ├─ Compiler.py
│  │  │  ├─ IR
│  │  │  │  ├─ IR.py
│  │  │  │  ├─ IRBuilderCSF.py
│  │  │  │  └─ IRUtil.py
│  │  │  ├─ Optimizations
│  │  │  │  ├─ GarbageCollector.py
│  │  │  │  └─ ReluMaxpoolOpti.py
│  │  │  ├─ README.md
│  │  │  ├─ SeeDot.py
│  │  │  ├─ Type.py
│  │  │  ├─ Util.py
│  │  │  └─ Writer.py
│  │  ├─ TFCompiler
│  │  │  ├─ DumpTFMtData.py
│  │  │  ├─ Graph.py
│  │  │  ├─ ProcessTFGraph.py
│  │  │  └─ TFNodesAST.py
│  │  ├─ TFEzPCLibrary
│  │  │  ├─ Library32_common.ezpc
│  │  │  ├─ Library32_cpp_post.ezpc
│  │  │  ├─ Library32_cpp_pre.ezpc
│  │  │  ├─ Library32_porthos.ezpc
│  │  │  ├─ Library32_sci.ezpc
│  │  │  ├─ Library64_common.ezpc
│  │  │  ├─ Library64_cpp_post.ezpc
│  │  │  ├─ Library64_cpp_pre.ezpc
│  │  │  ├─ Library64_porthos.ezpc
│  │  │  ├─ Library64_sci.ezpc
│  │  │  └─ concat_generator.ipynb
│  │  ├─ demos
│  │  │  └─ onnx
│  │  │     ├─ README.md
│  │  │     ├─ config.json
│  │  │     ├─ fetch_model.sh
│  │  │     ├─ input.jpg
│  │  │     ├─ pre_process.py
│  │  │     └─ run_onnx.py
│  │  ├─ node_git_adds.sh
│  │  └─ tests
│  │     ├─ conftest.py
│  │     ├─ onnx
│  │     │  └─ unittests
│  │     │     ├─ test_arith_binops.py
│  │     │     ├─ test_batchnorm.py
│  │     │     ├─ test_convolution.py
│  │     │     ├─ test_non_linear.py
│  │     │     ├─ test_shape_manipulation.py
│  │     │     └─ test_unaryops.py
│  │     ├─ pytest.ini
│  │     ├─ pytest_coverage_tf.config
│  │     ├─ tf
│  │     │  └─ unittests
│  │     │     ├─ test_arith_binops.py
│  │     │     ├─ test_batchnorm.py
│  │     │     ├─ test_convolution.py
│  │     │     ├─ test_non_linear.py
│  │     │     ├─ test_shape_manipulation.py
│  │     │     └─ test_unaryops.py
│  │     └─ utils.py
│  ├─ Beacon
│  │  ├─ README.md
│  │  ├─ beacon_frontend.py
│  │  ├─ beacon_frontend_conv.py
│  │  ├─ bench_client.sh
│  │  ├─ bench_server.sh
│  │  ├─ compile_conv_net.py
│  │  ├─ compile_networks.py
│  │  ├─ funcs.ezpc
│  │  ├─ micro_client.sh
│  │  └─ micro_server.sh
│  ├─ Dockerfile
│  ├─ Dockerfile_AI_Validation
│  ├─ EzPC
│  │  ├─ Docker
│  │  │  ├─ Dockerfile
│  │  │  ├─ compile_docker_example.sh
│  │  │  ├─ run_docker_example_client.sh
│  │  │  └─ run_docker_example_server.sh
│  │  ├─ EzPC
│  │  │  ├─ .ocamlinit
│  │  │  ├─ ABY_example
│  │  │  │  ├─ Makefile
│  │  │  │  ├─ common
│  │  │  │  │  ├─ ezpc.h
│  │  │  │  │  ├─ millionaire_prob.cpp
│  │  │  │  │  └─ millionaire_prob.h
│  │  │  │  └─ millionaire_prob_test.cpp
│  │  │  ├─ Library
│  │  │  │  ├─ Library32.ezpc
│  │  │  │  ├─ Library32_cppring.h
│  │  │  │  ├─ Library64.ezpc
│  │  │  │  └─ Library64_cppring.h
│  │  │  ├─ Makefile
│  │  │  ├─ OBLIVC_example
│  │  │  │  ├─ Makefile
│  │  │  │  ├─ ezpc.c
│  │  │  │  ├─ ezpc.h
│  │  │  │  └─ ezpc.oc
│  │  │  ├─ ast.ml
│  │  │  ├─ codegen.ml
│  │  │  ├─ codegenast.ml
│  │  │  ├─ codegencppfloat.ml
│  │  │  ├─ codegencppring.ml
│  │  │  ├─ codegenemp.ml
│  │  │  ├─ codegenfss.ml
│  │  │  ├─ codegenlib.ml
│  │  │  ├─ codegenoblivc.ml
│  │  │  ├─ codegenporthos.ml
│  │  │  ├─ codegensci.ml
│  │  │  ├─ codegensecfloat.ml
│  │  │  ├─ compile_aby.sh
│  │  │  ├─ compile_emp.sh
│  │  │  ├─ compile_secfloat.sh
│  │  │  ├─ config.ml
│  │  │  ├─ docker_test
│  │  │  │  ├─ docker_arith_example.ezpc
│  │  │  │  └─ docker_bin_example.ezpc
│  │  │  ├─ ezpc.sh
│  │  │  ├─ fssc
│  │  │  ├─ global.ml
│  │  │  ├─ infer.ml
│  │  │  ├─ lexer.mll
│  │  │  ├─ main.ml
│  │  │  ├─ optimizer.ml
│  │  │  ├─ parser.mly
│  │  │  ├─ partition.ml
│  │  │  ├─ resolvevars.ml
│  │  │  ├─ runemptests.sh
│  │  │  ├─ runtests.sh
│  │  │  ├─ secfloat.h
│  │  │  ├─ tc.ml
│  │  │  ├─ tcenv.ml
│  │  │  ├─ test_suite
│  │  │  │  ├─ binop.ezpc
│  │  │  │  ├─ decl.ezpc
│  │  │  │  ├─ dot_product.ezpc
│  │  │  │  ├─ for.ezpc
│  │  │  │  ├─ func.ezpc
│  │  │  │  ├─ if.ezpc
│  │  │  │  ├─ input.ezpc
│  │  │  │  ├─ output.ezpc
│  │  │  │  ├─ precompiled_output
│  │  │  │  │  ├─ binop0.cpp
│  │  │  │  │  ├─ decl0.cpp
│  │  │  │  │  ├─ dot_product0.cpp
│  │  │  │  │  ├─ for0.cpp
│  │  │  │  │  ├─ func0.cpp
│  │  │  │  │  ├─ if0.cpp
│  │  │  │  │  ├─ input0.cpp
│  │  │  │  │  ├─ output0.cpp
│  │  │  │  │  ├─ random_forest0.cpp
│  │  │  │  │  ├─ random_forest_polish0.cpp
│  │  │  │  │  └─ while0.cpp
│  │  │  │  ├─ precompiled_output_emp
│  │  │  │  │  ├─ binop0.cpp
│  │  │  │  │  ├─ decl0.cpp
│  │  │  │  │  ├─ dot_product0.cpp
│  │  │  │  │  ├─ for0.cpp
│  │  │  │  │  ├─ func0.cpp
│  │  │  │  │  ├─ if0.cpp
│  │  │  │  │  ├─ input0.cpp
│  │  │  │  │  ├─ output0.cpp
│  │  │  │  │  ├─ random_forest0.cpp
│  │  │  │  │  ├─ random_forest_polish0.cpp
│  │  │  │  │  └─ while0.cpp
│  │  │  │  ├─ random_forest.ezpc
│  │  │  │  ├─ random_forest_polish.ezpc
│  │  │  │  └─ while.ezpc
│  │  │  ├─ test_suite_float
│  │  │  │  ├─ binop_float.ezpc
│  │  │  │  ├─ decl.ezpc
│  │  │  │  ├─ dot_product_float.ezpc
│  │  │  │  ├─ for.ezpc
│  │  │  │  ├─ func.ezpc
│  │  │  │  ├─ if.ezpc
│  │  │  │  ├─ input.ezpc
│  │  │  │  ├─ output.ezpc
│  │  │  │  ├─ precompiled_output_emp
│  │  │  │  │  ├─ binop_float0.cpp
│  │  │  │  │  ├─ decl0.cpp
│  │  │  │  │  ├─ dot_product_float0.cpp
│  │  │  │  │  ├─ for0.cpp
│  │  │  │  │  ├─ func0.cpp
│  │  │  │  │  ├─ if0.cpp
│  │  │  │  │  ├─ input0.cpp
│  │  │  │  │  ├─ output0.cpp
│  │  │  │  │  └─ while0.cpp
│  │  │  │  ├─ precompiled_output_secfloat
│  │  │  │  │  ├─ binop_float0.cpp
│  │  │  │  │  ├─ decl0.cpp
│  │  │  │  │  ├─ dot_product_float0.cpp
│  │  │  │  │  ├─ for0.cpp
│  │  │  │  │  ├─ func0.cpp
│  │  │  │  │  ├─ if0.cpp
│  │  │  │  │  ├─ input0.cpp
│  │  │  │  │  ├─ output0.cpp
│  │  │  │  │  └─ while0.cpp
│  │  │  │  └─ while.ezpc
│  │  │  └─ utils.ml
│  │  └─ README.md
│  ├─ FSS
│  │  ├─ CMakeLists.txt
│  │  ├─ LICENSE
│  │  ├─ README.md
│  │  ├─ benchmarks
│  │  │  ├─ deepsecure.ezpc
│  │  │  ├─ google30.ezpc
│  │  │  ├─ heads1.ezpc
│  │  │  ├─ heads2.ezpc
│  │  │  ├─ heads3.ezpc
│  │  │  ├─ industrial.ezpc
│  │  │  ├─ input.inp
│  │  │  ├─ lstm.ezpc
│  │  │  ├─ minionn-cnn.ezpc
│  │  │  ├─ model.inp
│  │  │  ├─ resnet18.ezpc
│  │  │  └─ resnet50.ezpc
│  │  ├─ cmake
│  │  │  └─ FSSConfig.cmake.in
│  │  ├─ docker
│  │  │  ├─ Dockerfile
│  │  │  ├─ README.md
│  │  │  └─ setup_env_and_build.sh
│  │  ├─ microbenchmarks
│  │  │  ├─ invsqrt.ezpc
│  │  │  ├─ matmul_1.ezpc
│  │  │  ├─ matmul_2.ezpc
│  │  │  ├─ matmul_3.ezpc
│  │  │  ├─ sigmoid.ezpc
│  │  │  ├─ signextend.ezpc
│  │  │  ├─ tanh.ezpc
│  │  │  └─ truncatereduce.ezpc
│  │  ├─ src
│  │  │  ├─ ArgMapping.h
│  │  │  ├─ CMakeLists.txt
│  │  │  ├─ README.md
│  │  │  ├─ add.cpp
│  │  │  ├─ add.h
│  │  │  ├─ api.cpp
│  │  │  ├─ api.h
│  │  │  ├─ api_varied.cpp
│  │  │  ├─ api_varied.h
│  │  │  ├─ array.h
│  │  │  ├─ comms.cpp
│  │  │  ├─ comms.h
│  │  │  ├─ config.h
│  │  │  ├─ conv.cpp
│  │  │  ├─ conv.h
│  │  │  ├─ dcf.cpp
│  │  │  ├─ dcf.h
│  │  │  ├─ deps
│  │  │  │  └─ cryptoTools
│  │  │  │     ├─ LICENSE
│  │  │  │     └─ cryptoTools
│  │  │  │        ├─ Common
│  │  │  │        │  ├─ Defines.cpp
│  │  │  │        │  ├─ Defines.h
│  │  │  │        │  ├─ Log.cpp
│  │  │  │        │  ├─ Log.h
│  │  │  │        │  └─ config.h
│  │  │  │        ├─ Crypto
│  │  │  │        │  ├─ AES.cpp
│  │  │  │        │  ├─ AES.h
│  │  │  │        │  ├─ PRNG.cpp
│  │  │  │        │  └─ PRNG.h
│  │  │  │        └─ gsl
│  │  │  │           ├─ GSL.natvis
│  │  │  │           ├─ gls-lite.hpp
│  │  │  │           ├─ gsl
│  │  │  │           ├─ gsl_algorithm
│  │  │  │           ├─ gsl_assert
│  │  │  │           ├─ gsl_byte
│  │  │  │           ├─ gsl_util
│  │  │  │           ├─ multi_span
│  │  │  │           ├─ span
│  │  │  │           └─ string_span
│  │  │  ├─ fss.h
│  │  │  ├─ group_element.h
│  │  │  ├─ input_prng.cpp
│  │  │  ├─ input_prng.h
│  │  │  ├─ keypack.h
│  │  │  ├─ lib.cpp
│  │  │  ├─ lib.ezpc
│  │  │  ├─ lib.h
│  │  │  ├─ mult.cpp
│  │  │  ├─ mult.h
│  │  │  ├─ prng.cpp
│  │  │  ├─ pubdiv.cpp
│  │  │  ├─ pubdiv.h
│  │  │  ├─ spline.cpp
│  │  │  ├─ spline.h
│  │  │  ├─ utils.cpp
│  │  │  └─ utils.h
│  │  └─ tests
│  │     ├─ google30
│  │     │  ├─ input1.txt
│  │     │  ├─ input2.txt
│  │     │  ├─ main.ezpc
│  │     │  ├─ output1.txt
│  │     │  └─ output2.txt
│  │     ├─ run.sh
│  │     ├─ runall.sh
│  │     ├─ sigmoid
│  │     │  ├─ input1.txt
│  │     │  ├─ input2.txt
│  │     │  ├─ main.ezpc
│  │     │  ├─ output1.txt
│  │     │  └─ output2.txt
│  │     ├─ signextension
│  │     │  ├─ input1.txt
│  │     │  ├─ input2.txt
│  │     │  ├─ main.ezpc
│  │     │  ├─ output1.txt
│  │     │  └─ output2.txt
│  │     ├─ tanh
│  │     │  ├─ input1.txt
│  │     │  ├─ input2.txt
│  │     │  ├─ main.ezpc
│  │     │  ├─ output1.txt
│  │     │  └─ output2.txt
│  │     └─ truncatereduce
│  │        ├─ input1.txt
│  │        ├─ input2.txt
│  │        ├─ main.ezpc
│  │        ├─ output1.txt
│  │        └─ output2.txt
│  ├─ LICENSE
│  ├─ OnnxBridge
│  │  ├─ LLAMA
│  │  │  ├─ compile_llama.sh
│  │  │  ├─ sytorchBackendRep.py
│  │  │  └─ sytorch_func_calls.py
│  │  ├─ README.md
│  │  ├─ Secfloat
│  │  │  ├─ backendRep.py
│  │  │  ├─ compile_secfloat.sh
│  │  │  ├─ demo
│  │  │  │  ├─ Readme.md
│  │  │  │  ├─ fetch_image.sh
│  │  │  │  ├─ fetch_model.sh
│  │  │  │  ├─ mnist
│  │  │  │  │  └─ create_image.py
│  │  │  │  └─ process_image.sh
│  │  │  ├─ func_calls.py
│  │  │  ├─ lib_cleartext
│  │  │  │  ├─ cleartext_common.cpp
│  │  │  │  └─ cleartext_inout.cpp
│  │  │  └─ lib_secfloat
│  │  │     ├─ common.cpp
│  │  │     ├─ inout.cpp
│  │  │     └─ link_secfloat.cpp
│  │  ├─ backend.py
│  │  ├─ helper
│  │  │  ├─ compare_np_arrs.py
│  │  │  ├─ convert_np_to_float_inp.py
│  │  │  ├─ create_np_array.py
│  │  │  ├─ make_model.py
│  │  │  ├─ make_np_arr.py
│  │  │  ├─ pre_process.py
│  │  │  └─ run_onnx.py
│  │  ├─ main.py
│  │  ├─ requirements.txt
│  │  └─ utils
│  │     ├─ __init__.py
│  │     ├─ backend_helper.py
│  │     ├─ logger.py
│  │     ├─ nodes.py
│  │     ├─ onnx2IR_helper.py
│  │     ├─ onnx_nodes.py
│  │     └─ optimizations.py
│  ├─ Porthos
│  │  ├─ README.md
│  │  ├─ files
│  │  │  ├─ addresses
│  │  │  └─ keys
│  │  │     ├─ keyA
│  │  │     ├─ keyAB
│  │  │     ├─ keyB
│  │  │     └─ keyD
│  │  ├─ party0.sh
│  │  ├─ party1.sh
│  │  ├─ party2.sh
│  │  ├─ setup-eigen.sh
│  │  └─ src
│  │     ├─ AESObject.cpp
│  │     ├─ AESObject.h
│  │     ├─ CMakeLists.txt
│  │     ├─ EzPCFunctionalities.cpp
│  │     ├─ EzPCFunctionalities.h
│  │     ├─ Functionalities.cpp
│  │     ├─ Functionalities.h
│  │     ├─ ParallelAESObject.cpp
│  │     ├─ ParallelAESObject.h
│  │     ├─ basicSockets.cpp
│  │     ├─ basicSockets.h
│  │     ├─ connect.cpp
│  │     ├─ connect.h
│  │     ├─ example_neural_nets
│  │     │  ├─ mainDenseNet121.cpp
│  │     │  ├─ mainResNet50.cpp
│  │     │  ├─ mainSqNetImgNet.cpp
│  │     │  └─ network_config.h
│  │     ├─ ezpc.h
│  │     ├─ globals.h
│  │     ├─ secondary.cpp
│  │     ├─ secondary.h
│  │     ├─ tools.cpp
│  │     └─ tools.h
│  ├─ README.md
│  ├─ SCI
│  │  ├─ CMakeLists.txt
│  │  ├─ README.md
│  │  ├─ cmake
│  │  │  ├─ SCIConfig.cmake
│  │  │  ├─ SCIConfig.cmake.in
│  │  │  ├─ SecureFixedPointConfig.cmake
│  │  │  ├─ install_EMP.cmake
│  │  │  ├─ install_Eigen3.cmake
│  │  │  └─ seal.patch
│  │  ├─ extern
│  │  │  ├─ SEAL
│  │  │  └─ eigen
│  │  ├─ networks
│  │  │  ├─ CMakeLists.txt
│  │  │  ├─ inputs
│  │  │  │  ├─ ffnn_input128.inp
│  │  │  │  ├─ ffnn_labels128.inp
│  │  │  │  ├─ ffnn_weights.inp
│  │  │  │  ├─ hinet_input4.inp
│  │  │  │  ├─ hinet_labels4.inp
│  │  │  │  ├─ hinet_weights.inp
│  │  │  │  ├─ lenet_input128.inp
│  │  │  │  ├─ lenet_labels128.inp
│  │  │  │  ├─ lenet_weights.inp
│  │  │  │  ├─ logistic_input128.inp
│  │  │  │  ├─ logistic_labels128.inp
│  │  │  │  └─ logistic_weights.inp
│  │  │  ├─ main_densenet121.cpp
│  │  │  ├─ main_ffnn128.cpp
│  │  │  ├─ main_hinet4.cpp
│  │  │  ├─ main_lenet128.cpp
│  │  │  ├─ main_logistic128.cpp
│  │  │  ├─ main_minionn.cpp
│  │  │  ├─ main_relevance32.cpp
│  │  │  ├─ main_resnet32_cifar.cpp
│  │  │  ├─ main_resnet50.cpp
│  │  │  └─ main_sqnet.cpp
│  │  ├─ src
│  │  │  ├─ BuildingBlocks
│  │  │  │  ├─ CMakeLists.txt
│  │  │  │  ├─ aux-protocols.cpp
│  │  │  │  ├─ aux-protocols.h
│  │  │  │  ├─ truncation.cpp
│  │  │  │  ├─ truncation.h
│  │  │  │  ├─ value-extension.cpp
│  │  │  │  └─ value-extension.h
│  │  │  ├─ CMakeLists.txt
│  │  │  ├─ FloatingPoint
│  │  │  │  ├─ CMakeLists.txt
│  │  │  │  ├─ bool-data.cpp
│  │  │  │  ├─ bool-data.h
│  │  │  │  ├─ fixed-point.cpp
│  │  │  │  ├─ fixed-point.h
│  │  │  │  ├─ floating-point.cpp
│  │  │  │  ├─ floating-point.h
│  │  │  │  ├─ fp-math-coeffs.h
│  │  │  │  ├─ fp-math.cpp
│  │  │  │  └─ fp-math.h
│  │  │  ├─ GC
│  │  │  │  ├─ CMakeLists.txt
│  │  │  │  ├─ aes_opt.h
│  │  │  │  ├─ bit.h
│  │  │  │  ├─ bit.hpp
│  │  │  │  ├─ circuit_execution.h
│  │  │  │  ├─ comparable.h
│  │  │  │  ├─ emp-sh2pc.h
│  │  │  │  ├─ emp-tool.cpp
│  │  │  │  ├─ emp-tool.h
│  │  │  │  ├─ f2k.h
│  │  │  │  ├─ halfgate_eva.cpp
│  │  │  │  ├─ halfgate_eva.h
│  │  │  │  ├─ halfgate_gen.cpp
│  │  │  │  ├─ halfgate_gen.h
│  │  │  │  ├─ integer.h
│  │  │  │  ├─ integer.hpp
│  │  │  │  ├─ mitccrh.h
│  │  │  │  ├─ number.h
│  │  │  │  ├─ protocol_execution.h
│  │  │  │  ├─ semihonest.h
│  │  │  │  ├─ sh_eva.h
│  │  │  │  ├─ sh_gen.h
│  │  │  │  ├─ sh_party.h
│  │  │  │  ├─ swappable.h
│  │  │  │  └─ utils.h
│  │  │  ├─ LinearHE
│  │  │  │  ├─ CMakeLists.txt
│  │  │  │  ├─ conv-field.cpp
│  │  │  │  ├─ conv-field.h
│  │  │  │  ├─ defines-HE.h
│  │  │  │  ├─ elemwise-prod-field.cpp
│  │  │  │  ├─ elemwise-prod-field.h
│  │  │  │  ├─ fc-field.cpp
│  │  │  │  ├─ fc-field.h
│  │  │  │  ├─ generate_primes.py
│  │  │  │  ├─ utils-HE.cpp
│  │  │  │  └─ utils-HE.h
│  │  │  ├─ LinearOT
│  │  │  │  ├─ CMakeLists.txt
│  │  │  │  ├─ linear-ot.cpp
│  │  │  │  ├─ linear-ot.h
│  │  │  │  └─ linear-uniform.h
│  │  │  ├─ Math
│  │  │  │  ├─ CMakeLists.txt
│  │  │  │  ├─ math-functions.cpp
│  │  │  │  └─ math-functions.h
│  │  │  ├─ Millionaire
│  │  │  │  ├─ CMakeLists.txt
│  │  │  │  ├─ bit-triple-generator.h
│  │  │  │  ├─ equality.h
│  │  │  │  ├─ millionaire.h
│  │  │  │  └─ millionaire_with_equality.h
│  │  │  ├─ NonLinear
│  │  │  │  ├─ CMakeLists.txt
│  │  │  │  ├─ argmax.h
│  │  │  │  ├─ drelu-field.h
│  │  │  │  ├─ maxpool.h
│  │  │  │  ├─ relu-field.h
│  │  │  │  ├─ relu-interface.h
│  │  │  │  └─ relu-ring.h
│  │  │  ├─ OT
│  │  │  │  ├─ CMakeLists.txt
│  │  │  │  ├─ emp-ot.h
│  │  │  │  ├─ ideal.h
│  │  │  │  ├─ iknp.h
│  │  │  │  ├─ kkot.h
│  │  │  │  ├─ np.h
│  │  │  │  ├─ ot-utils.h
│  │  │  │  ├─ ot.h
│  │  │  │  ├─ ot_pack.h
│  │  │  │  ├─ split-iknp.h
│  │  │  │  ├─ split-kkot.h
│  │  │  │  └─ split-utils.h
│  │  │  ├─ cleartext_library_fixed.cpp
│  │  │  ├─ cleartext_library_fixed.h
│  │  │  ├─ cleartext_library_fixed_uniform.h
│  │  │  ├─ cleartext_library_float.cpp
│  │  │  ├─ cleartext_library_float.h
│  │  │  ├─ defines.h
│  │  │  ├─ defines_float.h
│  │  │  ├─ defines_uniform.h
│  │  │  ├─ functionalities_uniform.h
│  │  │  ├─ globals.cpp
│  │  │  ├─ globals.h
│  │  │  ├─ globals_float.cpp
│  │  │  ├─ globals_float.h
│  │  │  ├─ library_fixed.cpp
│  │  │  ├─ library_fixed.h
│  │  │  ├─ library_fixed_common.h
│  │  │  ├─ library_fixed_uniform.cpp
│  │  │  ├─ library_fixed_uniform.h
│  │  │  ├─ library_float.h
│  │  │  ├─ library_float_beacon.cpp
│  │  │  ├─ library_float_common.cpp
│  │  │  ├─ library_float_secfloat.cpp
│  │  │  └─ utils
│  │  │     ├─ ArgMapping
│  │  │     │  ├─ ArgMapping.h
│  │  │     │  ├─ LICENSE
│  │  │     │  └─ NOTICE
│  │  │     ├─ CMakeLists.txt
│  │  │     ├─ ThreadPool.h
│  │  │     ├─ aes-ni.h
│  │  │     ├─ aes.h
│  │  │     ├─ aes_opt.h
│  │  │     ├─ block.h
│  │  │     ├─ ccrf.h
│  │  │     ├─ ccrh.h
│  │  │     ├─ cmake
│  │  │     │  ├─ FindGMP.cmake
│  │  │     │  └─ source_of_randomness.cmake
│  │  │     ├─ constants.h
│  │  │     ├─ crh.h
│  │  │     ├─ emp-tool.h
│  │  │     ├─ f2k.h
│  │  │     ├─ group.h
│  │  │     ├─ group_openssl.h
│  │  │     ├─ hash.h
│  │  │     ├─ io_channel.h
│  │  │     ├─ io_pack.h
│  │  │     ├─ net_io_channel.h
│  │  │     ├─ prg.h
│  │  │     ├─ prp.h
│  │  │     ├─ sse2neon.h
│  │  │     ├─ tccrh.h
│  │  │     ├─ ubuntu_terminal_colors.h
│  │  │     ├─ utils.h
│  │  │     └─ utils.hpp
│  │  └─ tests
│  │     ├─ CMakeLists.txt
│  │     ├─ FindMPFR.cmake
│  │     ├─ activation
│  │     │  ├─ CMakeLists.txt
│  │     │  ├─ bolt_gelu.cpp
│  │     │  ├─ bolt_layer_norm.cpp
│  │     │  ├─ bolt_softmax.cpp
│  │     │  ├─ bolt_tanh.cpp
│  │     │  ├─ iron_gelu.cpp
│  │     │  ├─ iron_layer_norm.cpp
│  │     │  ├─ iron_softmax.cpp
│  │     │  └─ iron_tanh.cpp
│  │     ├─ bert_bolt
│  │     │  ├─ CMakeLists.txt
│  │     │  ├─ bert.cpp
│  │     │  ├─ bert.h
│  │     │  ├─ bert_utils.cpp
│  │     │  ├─ bert_utils.h
│  │     │  ├─ bolt_bert.cpp
│  │     │  ├─ bolt_bert_word_elimination.cpp
│  │     │  ├─ he.cpp
│  │     │  ├─ he.h
│  │     │  ├─ linear.cpp
│  │     │  ├─ linear.h
│  │     │  ├─ nonlinear.cpp
│  │     │  ├─ nonlinear.h
│  │     │  └─ test_nonlinear.cpp
│  │     ├─ bert_iron
│  │     │  ├─ CMakeLists.txt
│  │     │  ├─ bert.cpp
│  │     │  ├─ bert.h
│  │     │  ├─ bert_utils.cpp
│  │     │  ├─ bert_utils.h
│  │     │  ├─ he.cpp
│  │     │  ├─ he.h
│  │     │  ├─ iron_bert.cpp
│  │     │  ├─ linear.cpp
│  │     │  ├─ linear.h
│  │     │  ├─ nonlinear.cpp
│  │     │  ├─ nonlinear.h
│  │     │  └─ test_nonlinear.cpp
│  │     └─ moe_private
│  ├─ SIRNN
│  │  ├─ Library_SIRNN.ezpc
│  │  ├─ README.md
│  │  ├─ preProcessSIRNN.py
│  │  ├─ secureCodegen.py
│  │  └─ templates
│  │     ├─ CMakeLists.txt
│  │     ├─ CMakeLists.txt_Dataset
│  │     ├─ main.cpp
│  │     └─ predictors.h
│  ├─ bert.py
│  ├─ results
│  │  ├─ no_pruning
│  │  │  ├─ mrpc.txt
│  │  │  ├─ rte.txt
│  │  │  ├─ sst-2.txt
│  │  │  └─ sts-b.txt
│  │  └─ pruning
│  │     ├─ mrpc.txt
│  │     ├─ rte.txt
│  │     ├─ sst-2.txt
│  │     └─ sts-b.txt
│  ├─ setup_env_and_build.sh
│  └─ sytorch
│     ├─ CMakeLists.txt
│     ├─ README.md
│     ├─ Toy example- multiple inference.md
│     ├─ Toy example- single inference.md
│     ├─ ext
│     │  ├─ cryptoTools
│     │  │  ├─ CMakeLists.txt
│     │  │  ├─ LICENSE
│     │  │  └─ cryptoTools
│     │  │     ├─ Common
│     │  │     │  ├─ Defines.cpp
│     │  │     │  ├─ Defines.h
│     │  │     │  ├─ Log.cpp
│     │  │     │  ├─ Log.h
│     │  │     │  └─ config.h
│     │  │     ├─ Crypto
│     │  │     │  ├─ AES.cpp
│     │  │     │  ├─ AES.h
│     │  │     │  ├─ PRNG.cpp
│     │  │     │  └─ PRNG.h
│     │  │     └─ gsl
│     │  │        ├─ GSL.natvis
│     │  │        ├─ gls-lite.hpp
│     │  │        ├─ gsl
│     │  │        ├─ gsl_algorithm
│     │  │        ├─ gsl_assert
│     │  │        ├─ gsl_byte
│     │  │        ├─ gsl_util
│     │  │        ├─ multi_span
│     │  │        ├─ span
│     │  │        └─ string_span
│     │  └─ llama
│     │     ├─ CMakeLists.txt
│     │     ├─ and.cpp
│     │     ├─ and.h
│     │     ├─ api.cpp
│     │     ├─ conv.cpp
│     │     ├─ conv.h
│     │     ├─ dcf.cpp
│     │     ├─ dcf.h
│     │     ├─ include
│     │     │  └─ llama
│     │     │     ├─ api.h
│     │     │     ├─ array.h
│     │     │     ├─ assert.h
│     │     │     ├─ comms.h
│     │     │     ├─ config.h
│     │     │     ├─ freekey.h
│     │     │     ├─ group_element.h
│     │     │     ├─ input_prng.h
│     │     │     ├─ keypack.h
│     │     │     ├─ prng.h
│     │     │     ├─ stats.h
│     │     │     └─ utils.h
│     │     ├─ mult.cpp
│     │     ├─ mult.h
│     │     ├─ pubdiv.cpp
│     │     ├─ pubdiv.h
│     │     ├─ relu.cpp
│     │     ├─ relu.h
│     │     └─ src
│     │        └─ llama
│     │           ├─ comms.cpp
│     │           ├─ config.cpp
│     │           ├─ input_prng.cpp
│     │           ├─ prng.cpp
│     │           ├─ stats.cpp
│     │           └─ utils.cpp
│     ├─ ezpc-cli-2.sh
│     ├─ ezpc-cli.sh
│     ├─ include
│     │  └─ sytorch
│     │     ├─ backend
│     │     │  ├─ backend.h
│     │     │  ├─ cleartext.h
│     │     │  ├─ llama_base.h
│     │     │  └─ llama_extended.h
│     │     ├─ graph.h
│     │     ├─ layers
│     │     │  └─ layers.h
│     │     ├─ module.h
│     │     ├─ random.h
│     │     ├─ tensor.h
│     │     └─ utils.h
│     ├─ scripts
│     │  ├─ dealer.py
│     │  ├─ download_keys.py
│     │  └─ server.py
│     └─ src
│        └─ sytorch
│           ├─ backend
│           │  └─ cleartext.cpp
│           └─ random.cpp
├─ LLM-Adapters
│  ├─ DATA_LICENSE
│  ├─ LICENSE
│  ├─ README.md
│  ├─ commonsense_evaluate.py
│  ├─ dataset
│  │  ├─ AQuA
│  │  │  ├─ AQuA.json
│  │  │  ├─ aqua_1.json
│  │  │  └─ test.json
│  │  ├─ ARC-Challenge
│  │  │  ├─ test.json
│  │  │  └─ train.json
│  │  ├─ ARC-Easy
│  │  │  ├─ test.json
│  │  │  └─ train.json
│  │  ├─ AddSub
│  │  │  ├─ AddSub.json
│  │  │  ├─ addsub_1.json
│  │  │  └─ test.json
│  │  ├─ MultiArith
│  │  │  ├─ MultiArith.json
│  │  │  ├─ multiarith_1.json
│  │  │  └─ test.json
│  │  ├─ SVAMP
│  │  │  ├─ SVAMP.json
│  │  │  ├─ svamp_1.json
│  │  │  └─ test.json
│  │  ├─ SingleEq
│  │  │  ├─ SingleEq.json
│  │  │  ├─ singleeq_1.json
│  │  │  └─ test.json
│  │  ├─ boolq
│  │  │  ├─ test.json
│  │  │  └─ train.json
│  │  ├─ gsm8k
│  │  │  ├─ gsm8k.json
│  │  │  ├─ gsm8k_1.json
│  │  │  └─ test.json
│  │  ├─ hellaswag
│  │  │  ├─ test.json
│  │  │  └─ train.json
│  │  ├─ mathqa
│  │  │  └─ test.json
│  │  ├─ mawps
│  │  │  ├─ data_process.py
│  │  │  ├─ test.json
│  │  │  ├─ testset.json
│  │  │  ├─ trainset.json
│  │  │  └─ validset.json
│  │  ├─ openbookqa
│  │  │  ├─ test.json
│  │  │  └─ train.json
│  │  ├─ piqa
│  │  │  ├─ test.json
│  │  │  └─ train.json
│  │  ├─ social_i_qa
│  │  │  ├─ test.json
│  │  │  └─ train.json
│  │  └─ winogrande
│  │     ├─ test.json
│  │     └─ train.json
│  ├─ evaluate.py
│  ├─ export_hf_checkpoint.py
│  ├─ export_state_dict_checkpoint.py
│  ├─ finetune.py
│  ├─ ft-training_set
│  │  ├─ alpaca_data.json
│  │  ├─ alpaca_data_cleaned.json
│  │  ├─ commonsense_15k.json
│  │  ├─ commonsense_170k.json
│  │  ├─ math_10k.json
│  │  ├─ math_14k.json
│  │  ├─ math_50k.json
│  │  ├─ math_7k.json
│  │  └─ math_data.json
│  ├─ generate.py
│  ├─ lengths.ipynb
│  ├─ math_running_commands
│  ├─ mathqa.py
│  ├─ multi_dataset_eval.py
│  ├─ peft
│  │  ├─ LICENSE
│  │  ├─ Makefile
│  │  ├─ pyproject.toml
│  │  ├─ setup.py
│  │  ├─ src
│  │  │  └─ peft
│  │  │     ├─ __init__.py
│  │  │     ├─ __pycache__
│  │  │     │  ├─ __init__.cpython-39.pyc
│  │  │     │  ├─ mapping.cpython-39.pyc
│  │  │     │  └─ peft_model.cpython-39.pyc
│  │  │     ├─ mapping.py
│  │  │     ├─ peft_model.py
│  │  │     ├─ tuners
│  │  │     │  ├─ __init__.py
│  │  │     │  ├─ __pycache__
│  │  │     │  │  ├─ __init__.cpython-39.pyc
│  │  │     │  │  ├─ bottleneck.cpython-39.pyc
│  │  │     │  │  ├─ lora.cpython-39.pyc
│  │  │     │  │  ├─ p_tuning.cpython-39.pyc
│  │  │     │  │  ├─ prefix_tuning.cpython-39.pyc
│  │  │     │  │  └─ prompt_tuning.cpython-39.pyc
│  │  │     │  ├─ bottleneck.py
│  │  │     │  ├─ lora.py
│  │  │     │  ├─ p_tuning.py
│  │  │     │  ├─ prefix_tuning.py
│  │  │     │  └─ prompt_tuning.py
│  │  │     └─ utils
│  │  │        ├─ __init__.py
│  │  │        ├─ __pycache__
│  │  │        │  ├─ __init__.cpython-39.pyc
│  │  │        │  ├─ adapters_utils.cpython-39.pyc
│  │  │        │  ├─ config.cpython-39.pyc
│  │  │        │  ├─ other.cpython-39.pyc
│  │  │        │  └─ save_and_load.cpython-39.pyc
│  │  │        ├─ adapters_utils.py
│  │  │        ├─ config.py
│  │  │        ├─ other.py
│  │  │        └─ save_and_load.py
│  │  └─ tests
│  │     ├─ __init__.py
│  │     ├─ test_config.py
│  │     ├─ test_peft_model.py
│  │     ├─ testing_common.py
│  │     └─ testing_utils.py
│  ├─ picture.jpg
│  ├─ pyproject.toml
│  └─ requirements.txt
├─ README.md
├─ __pycache__
│  ├─ function_handler.cpython-310.pyc
│  ├─ layer_importance_evaluator.cpython-310.pyc
│  └─ layer_importance_evaluator_old.cpython-310.pyc
├─ approximation.py
├─ bert-test.py
├─ commonsense_170k.json
├─ commonsense_evaluate.py
├─ function_handler.py
├─ importance-aware-sparse-tuning-IST-paper
│  ├─ DATA_LICENSE
│  ├─ LICENSE
│  ├─ README.md
│  ├─ __pycache__
│  │  └─ ist.cpython-310.pyc
│  ├─ commonsense_170k.json
│  ├─ commonsense_evaluate.py
│  ├─ dataset
│  │  ├─ AQuA
│  │  │  ├─ AQuA.json
│  │  │  ├─ aqua_1.json
│  │  │  └─ test.json
│  │  ├─ ARC-Challenge
│  │  │  ├─ test.json
│  │  │  └─ train.json
│  │  ├─ ARC-Easy
│  │  │  ├─ test.json
│  │  │  └─ train.json
│  │  ├─ AddSub
│  │  │  ├─ AddSub.json
│  │  │  ├─ addsub_1.json
│  │  │  └─ test.json
│  │  ├─ MultiArith
│  │  │  ├─ MultiArith.json
│  │  │  ├─ multiarith_1.json
│  │  │  └─ test.json
│  │  ├─ SVAMP
│  │  │  ├─ SVAMP.json
│  │  │  ├─ svamp_1.json
│  │  │  └─ test.json
│  │  ├─ SingleEq
│  │  │  ├─ SingleEq.json
│  │  │  ├─ singleeq_1.json
│  │  │  └─ test.json
│  │  ├─ boolq
│  │  │  ├─ test.json
│  │  │  └─ train.json
│  │  ├─ gsm8k
│  │  │  ├─ gsm8k.json
│  │  │  ├─ gsm8k_1.json
│  │  │  └─ test.json
│  │  ├─ hellaswag
│  │  │  ├─ test.json
│  │  │  └─ train.json
│  │  ├─ mathqa
│  │  │  └─ test.json
│  │  ├─ mawps
│  │  │  ├─ data_process.py
│  │  │  ├─ test.json
│  │  │  ├─ testset.json
│  │  │  ├─ trainset.json
│  │  │  └─ validset.json
│  │  ├─ openbookqa
│  │  │  ├─ test.json
│  │  │  └─ train.json
│  │  ├─ piqa
│  │  │  ├─ test.json
│  │  │  └─ train.json
│  │  ├─ social_i_qa
│  │  │  ├─ test.json
│  │  │  └─ train.json
│  │  └─ winogrande
│  │     ├─ test.json
│  │     └─ train.json
│  ├─ experiment
│  ├─ finetune.py
│  ├─ finetuned_result
│  │  └─ r32_lr2e-4
│  ├─ generate.py
│  ├─ ist.py
│  ├─ llama3_8B_DoRA_IST.sh
│  ├─ llama_7B_LoRA_IST.sh
│  ├─ llama_7B_LoRA_RST.sh
│  ├─ peft
│  │  ├─ LICENSE
│  │  ├─ Makefile
│  │  ├─ pyproject.toml
│  │  ├─ setup.py
│  │  ├─ src
│  │  │  ├─ peft
│  │  │  │  ├─ __init__.py
│  │  │  │  ├─ __pycache__
│  │  │  │  │  ├─ __init__.cpython-310.pyc
│  │  │  │  │  ├─ mapping.cpython-310.pyc
│  │  │  │  │  └─ peft_model.cpython-310.pyc
│  │  │  │  ├─ mapping.py
│  │  │  │  ├─ peft_model.py
│  │  │  │  ├─ tuners
│  │  │  │  │  ├─ __init__.py
│  │  │  │  │  ├─ __pycache__
│  │  │  │  │  │  ├─ __init__.cpython-310.pyc
│  │  │  │  │  │  ├─ bottleneck.cpython-310.pyc
│  │  │  │  │  │  ├─ dora.cpython-310.pyc
│  │  │  │  │  │  ├─ lora.cpython-310.pyc
│  │  │  │  │  │  ├─ p_tuning.cpython-310.pyc
│  │  │  │  │  │  ├─ prefix_tuning.cpython-310.pyc
│  │  │  │  │  │  └─ prompt_tuning.cpython-310.pyc
│  │  │  │  │  ├─ bottleneck.py
│  │  │  │  │  ├─ dora.py
│  │  │  │  │  ├─ lora.py
│  │  │  │  │  ├─ p_tuning.py
│  │  │  │  │  ├─ prefix_tuning.py
│  │  │  │  │  ├─ prefix_tuning_back.py
│  │  │  │  │  └─ prompt_tuning.py
│  │  │  │  └─ utils
│  │  │  │     ├─ __init__.py
│  │  │  │     ├─ __pycache__
│  │  │  │     │  ├─ __init__.cpython-310.pyc
│  │  │  │     │  ├─ adapters_utils.cpython-310.pyc
│  │  │  │     │  ├─ config.cpython-310.pyc
│  │  │  │     │  ├─ other.cpython-310.pyc
│  │  │  │     │  └─ save_and_load.cpython-310.pyc
│  │  │  │     ├─ adapters_utils.py
│  │  │  │     ├─ config.py
│  │  │  │     ├─ other.py
│  │  │  │     └─ save_and_load.py
│  │  │  └─ peft.egg-info
│  │  │     ├─ PKG-INFO
│  │  │     ├─ SOURCES.txt
│  │  │     ├─ dependency_links.txt
│  │  │     ├─ requires.txt
│  │  │     └─ top_level.txt
│  │  └─ tests
│  │     ├─ __init__.py
│  │     ├─ test_config.py
│  │     ├─ test_peft_model.py
│  │     ├─ testing_common.py
│  │     └─ testing_utils.py
│  └─ requirements.txt
├─ importance_scores_lr20_steps20_degree2_actions24_modelBertForSequenceClassification.txt
├─ inference_output
├─ layer_importance_evaluator.py
├─ layer_importance_evaluator_mrpc.py
├─ layer_importance_evaluator_nonegroup.py
├─ llama_7B_LayerImportance.sh
├─ moe_sample.py
├─ output.log
├─ qk_v_spectral_norm_tables.xlsx
├─ results
└─ rl_tune.py

```