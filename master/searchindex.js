Search.setIndex({docnames:["README","configuration","data","index","package/modules","package/opennmt","package/opennmt.config","package/opennmt.constants","package/opennmt.decoders","package/opennmt.decoders.decoder","package/opennmt.decoders.rnn_decoder","package/opennmt.decoders.self_attention_decoder","package/opennmt.encoders","package/opennmt.encoders.conv_encoder","package/opennmt.encoders.encoder","package/opennmt.encoders.mean_encoder","package/opennmt.encoders.rnn_encoder","package/opennmt.encoders.self_attention_encoder","package/opennmt.inputters","package/opennmt.inputters.inputter","package/opennmt.inputters.record_inputter","package/opennmt.inputters.text_inputter","package/opennmt.models","package/opennmt.models.model","package/opennmt.models.sequence_classifier","package/opennmt.models.sequence_tagger","package/opennmt.models.sequence_to_sequence","package/opennmt.models.transformer","package/opennmt.tokenizers","package/opennmt.tokenizers.tokenizer","package/opennmt.utils","package/opennmt.utils.beam_search","package/opennmt.utils.bridge","package/opennmt.utils.cell","package/opennmt.utils.decay","package/opennmt.utils.evaluator","package/opennmt.utils.hooks","package/opennmt.utils.losses","package/opennmt.utils.misc","package/opennmt.utils.position","package/opennmt.utils.reducer","package/opennmt.utils.transformer","package/opennmt.utils.vocab","serving","training"],envversion:53,filenames:["README.md","configuration.md","data.md","index.rst","package/modules.rst","package/opennmt.rst","package/opennmt.config.rst","package/opennmt.constants.rst","package/opennmt.decoders.rst","package/opennmt.decoders.decoder.rst","package/opennmt.decoders.rnn_decoder.rst","package/opennmt.decoders.self_attention_decoder.rst","package/opennmt.encoders.rst","package/opennmt.encoders.conv_encoder.rst","package/opennmt.encoders.encoder.rst","package/opennmt.encoders.mean_encoder.rst","package/opennmt.encoders.rnn_encoder.rst","package/opennmt.encoders.self_attention_encoder.rst","package/opennmt.inputters.rst","package/opennmt.inputters.inputter.rst","package/opennmt.inputters.record_inputter.rst","package/opennmt.inputters.text_inputter.rst","package/opennmt.models.rst","package/opennmt.models.model.rst","package/opennmt.models.sequence_classifier.rst","package/opennmt.models.sequence_tagger.rst","package/opennmt.models.sequence_to_sequence.rst","package/opennmt.models.transformer.rst","package/opennmt.tokenizers.rst","package/opennmt.tokenizers.tokenizer.rst","package/opennmt.utils.rst","package/opennmt.utils.beam_search.rst","package/opennmt.utils.bridge.rst","package/opennmt.utils.cell.rst","package/opennmt.utils.decay.rst","package/opennmt.utils.evaluator.rst","package/opennmt.utils.hooks.rst","package/opennmt.utils.losses.rst","package/opennmt.utils.misc.rst","package/opennmt.utils.position.rst","package/opennmt.utils.reducer.rst","package/opennmt.utils.transformer.rst","package/opennmt.utils.vocab.rst","serving.md","training.md"],objects:{"":{opennmt:[5,0,0,"-"]},"opennmt.config":{load_config:[6,1,1,""],load_model_module:[6,1,1,""]},"opennmt.decoders":{decoder:[9,0,0,"-"],rnn_decoder:[10,0,0,"-"],self_attention_decoder:[11,0,0,"-"]},"opennmt.decoders.decoder":{Decoder:[9,2,1,""],get_embedding_fn:[9,1,1,""],get_sampling_probability:[9,1,1,""],logits_to_cum_log_probs:[9,1,1,""]},"opennmt.decoders.decoder.Decoder":{decode:[9,3,1,""],dynamic_decode:[9,3,1,""],dynamic_decode_and_search:[9,3,1,""]},"opennmt.decoders.rnn_decoder":{AttentionalRNNDecoder:[10,2,1,""],MultiAttentionalRNNDecoder:[10,2,1,""],RNNDecoder:[10,2,1,""]},"opennmt.decoders.rnn_decoder.AttentionalRNNDecoder":{__init__:[10,3,1,""]},"opennmt.decoders.rnn_decoder.MultiAttentionalRNNDecoder":{__init__:[10,3,1,""]},"opennmt.decoders.rnn_decoder.RNNDecoder":{__init__:[10,3,1,""],decode:[10,3,1,""],dynamic_decode:[10,3,1,""],dynamic_decode_and_search:[10,3,1,""]},"opennmt.decoders.self_attention_decoder":{SelfAttentionDecoder:[11,2,1,""]},"opennmt.decoders.self_attention_decoder.SelfAttentionDecoder":{__init__:[11,3,1,""],decode:[11,3,1,""],dynamic_decode:[11,3,1,""],dynamic_decode_and_search:[11,3,1,""]},"opennmt.encoders":{conv_encoder:[13,0,0,"-"],encoder:[14,0,0,"-"],mean_encoder:[15,0,0,"-"],rnn_encoder:[16,0,0,"-"],self_attention_encoder:[17,0,0,"-"]},"opennmt.encoders.conv_encoder":{ConvEncoder:[13,2,1,""]},"opennmt.encoders.conv_encoder.ConvEncoder":{__init__:[13,3,1,""],encode:[13,3,1,""]},"opennmt.encoders.encoder":{Encoder:[14,2,1,""],ParallelEncoder:[14,2,1,""],SequentialEncoder:[14,2,1,""]},"opennmt.encoders.encoder.Encoder":{encode:[14,3,1,""]},"opennmt.encoders.encoder.ParallelEncoder":{__init__:[14,3,1,""],encode:[14,3,1,""]},"opennmt.encoders.encoder.SequentialEncoder":{__init__:[14,3,1,""],encode:[14,3,1,""]},"opennmt.encoders.mean_encoder":{MeanEncoder:[15,2,1,""]},"opennmt.encoders.mean_encoder.MeanEncoder":{encode:[15,3,1,""]},"opennmt.encoders.rnn_encoder":{BidirectionalRNNEncoder:[16,2,1,""],GoogleRNNEncoder:[16,2,1,""],PyramidalRNNEncoder:[16,2,1,""],RNNEncoder:[16,2,1,""],UnidirectionalRNNEncoder:[16,2,1,""]},"opennmt.encoders.rnn_encoder.BidirectionalRNNEncoder":{__init__:[16,3,1,""],encode:[16,3,1,""]},"opennmt.encoders.rnn_encoder.GoogleRNNEncoder":{__init__:[16,3,1,""],encode:[16,3,1,""]},"opennmt.encoders.rnn_encoder.PyramidalRNNEncoder":{__init__:[16,3,1,""],encode:[16,3,1,""]},"opennmt.encoders.rnn_encoder.RNNEncoder":{__init__:[16,3,1,""],encode:[16,3,1,""]},"opennmt.encoders.rnn_encoder.UnidirectionalRNNEncoder":{__init__:[16,3,1,""],encode:[16,3,1,""]},"opennmt.encoders.self_attention_encoder":{SelfAttentionEncoder:[17,2,1,""]},"opennmt.encoders.self_attention_encoder.SelfAttentionEncoder":{__init__:[17,3,1,""],encode:[17,3,1,""]},"opennmt.inputters":{inputter:[19,0,0,"-"],record_inputter:[20,0,0,"-"],text_inputter:[21,0,0,"-"]},"opennmt.inputters.inputter":{Inputter:[19,2,1,""],MixedInputter:[19,2,1,""],MultiInputter:[19,2,1,""],ParallelInputter:[19,2,1,""]},"opennmt.inputters.inputter.Inputter":{add_process_hooks:[19,3,1,""],get_length:[19,3,1,""],get_serving_input_receiver:[19,3,1,""],initialize:[19,3,1,""],make_dataset:[19,3,1,""],process:[19,3,1,""],remove_data_field:[19,3,1,""],set_data_field:[19,3,1,""],transform:[19,3,1,""],transform_data:[19,3,1,""],visualize:[19,3,1,""]},"opennmt.inputters.inputter.MixedInputter":{__init__:[19,3,1,""],get_length:[19,3,1,""],make_dataset:[19,3,1,""],transform:[19,3,1,""]},"opennmt.inputters.inputter.MultiInputter":{initialize:[19,3,1,""],make_dataset:[19,3,1,""],transform:[19,3,1,""],visualize:[19,3,1,""]},"opennmt.inputters.inputter.ParallelInputter":{__init__:[19,3,1,""],get_length:[19,3,1,""],make_dataset:[19,3,1,""],transform:[19,3,1,""]},"opennmt.inputters.record_inputter":{SequenceRecordInputter:[20,2,1,""]},"opennmt.inputters.record_inputter.SequenceRecordInputter":{__init__:[20,3,1,""],get_length:[20,3,1,""],initialize:[20,3,1,""],make_dataset:[20,3,1,""],transform:[20,3,1,""]},"opennmt.inputters.text_inputter":{CharConvEmbedder:[21,2,1,""],TextInputter:[21,2,1,""],WordEmbedder:[21,2,1,""],load_pretrained_embeddings:[21,1,1,""],tokens_to_chars:[21,1,1,""],visualize_embeddings:[21,1,1,""]},"opennmt.inputters.text_inputter.CharConvEmbedder":{__init__:[21,3,1,""],initialize:[21,3,1,""],transform:[21,3,1,""],visualize:[21,3,1,""]},"opennmt.inputters.text_inputter.TextInputter":{get_length:[21,3,1,""],initialize:[21,3,1,""],make_dataset:[21,3,1,""],transform:[21,3,1,""]},"opennmt.inputters.text_inputter.WordEmbedder":{__init__:[21,3,1,""],initialize:[21,3,1,""],transform:[21,3,1,""],visualize:[21,3,1,""]},"opennmt.models":{model:[23,0,0,"-"],sequence_classifier:[24,0,0,"-"],sequence_tagger:[25,0,0,"-"],sequence_to_sequence:[26,0,0,"-"],transformer:[27,0,0,"-"]},"opennmt.models.model":{Model:[23,2,1,""],get_optimizer_class:[23,1,1,""],learning_rate_decay_fn:[23,1,1,""]},"opennmt.models.model.Model":{__call__:[23,3,1,""],input_fn:[23,3,1,""],print_prediction:[23,3,1,""],serving_input_fn:[23,3,1,""]},"opennmt.models.sequence_classifier":{SequenceClassifier:[24,2,1,""]},"opennmt.models.sequence_classifier.SequenceClassifier":{__init__:[24,3,1,""],print_prediction:[24,3,1,""]},"opennmt.models.sequence_tagger":{SequenceTagger:[25,2,1,""],flag_bioes_tags:[25,1,1,""]},"opennmt.models.sequence_tagger.SequenceTagger":{__init__:[25,3,1,""],print_prediction:[25,3,1,""]},"opennmt.models.sequence_to_sequence":{SequenceToSequence:[26,2,1,""],shift_target_sequence:[26,1,1,""]},"opennmt.models.sequence_to_sequence.SequenceToSequence":{__init__:[26,3,1,""],print_prediction:[26,3,1,""]},"opennmt.models.transformer":{Transformer:[27,2,1,""]},"opennmt.models.transformer.Transformer":{__init__:[27,3,1,""]},"opennmt.tokenizers":{tokenizer:[29,0,0,"-"]},"opennmt.tokenizers.tokenizer":{CharacterTokenizer:[29,2,1,""],SpaceTokenizer:[29,2,1,""],Tokenizer:[29,2,1,""]},"opennmt.tokenizers.tokenizer.Tokenizer":{__call__:[29,3,1,""],initialize:[29,3,1,""]},"opennmt.utils":{beam_search:[31,0,0,"-"],bridge:[32,0,0,"-"],cell:[33,0,0,"-"],decay:[34,0,0,"-"],evaluator:[35,0,0,"-"],hooks:[36,0,0,"-"],losses:[37,0,0,"-"],misc:[38,0,0,"-"],position:[39,0,0,"-"],reducer:[40,0,0,"-"],transformer:[41,0,0,"-"],vocab:[42,0,0,"-"]},"opennmt.utils.beam_search":{beam_search:[31,1,1,""],compute_batch_indices:[31,1,1,""],compute_topk_scores_and_seq:[31,1,1,""]},"opennmt.utils.bridge":{Bridge:[32,2,1,""],CopyBridge:[32,2,1,""],DenseBridge:[32,2,1,""],ZeroBridge:[32,2,1,""],assert_state_is_compatible:[32,1,1,""]},"opennmt.utils.bridge.Bridge":{__call__:[32,3,1,""]},"opennmt.utils.bridge.DenseBridge":{__init__:[32,3,1,""]},"opennmt.utils.cell":{build_cell:[33,1,1,""]},"opennmt.utils.decay":{noam_decay:[34,1,1,""]},"opennmt.utils.evaluator":{BLEUEvaluator:[35,2,1,""],ExternalEvaluator:[35,2,1,""],external_evaluation_fn:[35,1,1,""]},"opennmt.utils.evaluator.BLEUEvaluator":{name:[35,3,1,""],score:[35,3,1,""]},"opennmt.utils.evaluator.ExternalEvaluator":{__call__:[35,3,1,""],name:[35,4,1,""],score:[35,3,1,""]},"opennmt.utils.hooks":{CountersHook:[36,2,1,""],LogParametersCountHook:[36,2,1,""],SaveEvaluationPredictionHook:[36,2,1,""]},"opennmt.utils.hooks.CountersHook":{after_run:[36,3,1,""],before_run:[36,3,1,""],begin:[36,3,1,""]},"opennmt.utils.hooks.LogParametersCountHook":{begin:[36,3,1,""]},"opennmt.utils.hooks.SaveEvaluationPredictionHook":{__init__:[36,3,1,""],after_run:[36,3,1,""],before_run:[36,3,1,""],begin:[36,3,1,""],end:[36,3,1,""]},"opennmt.utils.losses":{cross_entropy_loss:[37,1,1,""],cross_entropy_sequence_loss:[37,1,1,""]},"opennmt.utils.misc":{add_dict_to_collection:[38,1,1,""],count_lines:[38,1,1,""],count_parameters:[38,1,1,""],extract_batches:[38,1,1,""],extract_prefixed_keys:[38,1,1,""],get_classnames_in_module:[38,1,1,""],get_dict_from_collection:[38,1,1,""],item_or_tuple:[38,1,1,""],print_bytes:[38,1,1,""]},"opennmt.utils.position":{PositionEmbedder:[39,2,1,""],PositionEncoder:[39,2,1,""],make_positions:[39,1,1,""]},"opennmt.utils.position.PositionEmbedder":{__init__:[39,3,1,""],encode:[39,3,1,""]},"opennmt.utils.position.PositionEncoder":{__call__:[39,3,1,""],apply:[39,3,1,""],apply_one:[39,3,1,""],encode:[39,3,1,""],encode_sequence:[39,3,1,""]},"opennmt.utils.reducer":{ConcatReducer:[40,2,1,""],JoinReducer:[40,2,1,""],MultiplyReducer:[40,2,1,""],Reducer:[40,2,1,""],SumReducer:[40,2,1,""],pad_in_time:[40,1,1,""],pad_n_with_identity:[40,1,1,""],pad_with_identity:[40,1,1,""],roll_sequence:[40,1,1,""]},"opennmt.utils.reducer.ConcatReducer":{reduce:[40,3,1,""],reduce_sequence:[40,3,1,""]},"opennmt.utils.reducer.JoinReducer":{reduce:[40,3,1,""],reduce_sequence:[40,3,1,""]},"opennmt.utils.reducer.MultiplyReducer":{reduce:[40,3,1,""],reduce_sequence:[40,3,1,""]},"opennmt.utils.reducer.Reducer":{reduce:[40,3,1,""],reduce_sequence:[40,3,1,""],zip_and_reduce:[40,3,1,""]},"opennmt.utils.reducer.SumReducer":{reduce:[40,3,1,""],reduce_sequence:[40,3,1,""]},"opennmt.utils.transformer":{build_future_mask:[41,1,1,""],build_sequence_mask:[41,1,1,""],combine_heads:[41,1,1,""],drop_and_add:[41,1,1,""],feed_forward:[41,1,1,""],multi_head_attention:[41,1,1,""],norm:[41,1,1,""],scaled_dot_attention:[41,1,1,""],split_heads:[41,1,1,""],tile_sequence_length:[41,1,1,""]},"opennmt.utils.vocab":{Vocab:[42,2,1,""]},"opennmt.utils.vocab.Vocab":{__init__:[42,3,1,""],add:[42,3,1,""],add_from_text:[42,3,1,""],lookup:[42,3,1,""],prune:[42,3,1,""],serialize:[42,3,1,""],size:[42,4,1,""]},opennmt:{config:[6,0,0,"-"],constants:[7,0,0,"-"],decoders:[8,0,0,"-"],encoders:[12,0,0,"-"],inputters:[18,0,0,"-"],models:[22,0,0,"-"],tokenizers:[28,0,0,"-"],utils:[30,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","method","Python method"],"4":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:method","4":"py:attribute"},terms:{"abstract":21,"byte":38,"case":[10,21,24],"class":[9,10,11,13,14,15,16,17,19,20,21,23,24,25,26,27,29,32,33,35,36,39,40,42],"default":[1,31,32,42],"export":[19,23,43],"final":[10,31],"float":41,"function":[6,9,19,21,23,25,31,32,33,34,35,38,40,41,43],"int":31,"new":[31,42],"null":2,"return":[6,9,10,14,16,19,21,23,25,26,29,31,32,33,34,35,37,38,39,40,41,42],"static":[20,40],"true":[10,16,19,21,23,25,33,35,37,40],"void":2,"while":31,EOS:31,For:[1,2,19,25,44],Ids:31,THE:21,The:[1,2,6,9,10,11,13,14,16,17,19,20,21,23,24,25,26,27,29,31,32,33,34,35,36,37,38,39,40,41,42,44],Then:44,These:31,Yes:2,__call__:[23,29,32,35,39],__init__:[10,11,13,14,16,17,19,20,21,24,25,26,27,32,36,39,42],_topk_:31,_topk_flag:31,_topk_scor:31,_topk_seq:31,about:[1,2,21,23,31],abs:[9,11,13,16,17,27,31,34,41],accept:[1,2],accordingli:[2,34],activ:[11,17,19,21,27,32,44],actual:1,adam_with_decai:1,add:[1,10,19,25,31,33,38,41,42],add_dict_to_collect:38,add_from_text:42,add_process_hook:19,add_to_collect:29,added:[0,10,16,19,26,33],addit:[19,21,23,25,29],addition:[10,26],advanc:[2,3],aerob:2,after:[10,16,19,23,25,31,33],after_run:36,against:[21,35],align:[2,21,25],aliv:31,all:[2,10,14,19,38,40,43],allow:1,alpha:31,also:[2,19,44],altern:10,ani:[21,29],antich:2,api:43,apidoc:0,apo:2,appli:[11,13,17,19,21,23,27,32,39,41],apply_on:39,arbitrarili:1,architectur:1,argument:[10,16,23,33,34,36],ark:2,ark_to_record:2,arrai:[21,25,38],arxiv:[9,11,13,16,17,27,31,34,41],assert:32,assert_state_is_compat:32,asset:[29,43],asset_filepath:29,assign:[19,21],associ:[21,42],assum:[19,21,39],assumpt:31,asynchron:44,attach:21,attend:[10,41],attent:[1,10,11,17,27,33,41],attention_dropout:[11,17,27],attention_lay:[10,33],attention_mechan:33,attention_mechanism_class:10,attention_wrapp:10,attentionalrnndecod:10,attentionmechan:[10,33],automat:[0,43],avail:[1,43],averag:24,axi:40,base:[9,10,11,13,14,15,16,17,19,20,21,23,24,25,26,27,29,31,32,35,36,39,40,42],basic:10,batch:[23,31,38,40],batch_po:31,batch_siz:[23,31,40,41],beam:[9,31],beam_search:[5,30],beam_siz:31,beam_width:[9,10,11],beat:2,becaus:[31,43],been:[2,25],befor:10,before_run:36,begin:36,behavior:21,being:31,beyond:2,bidirect:16,bidirectionalrnnencod:16,bin:[1,2,44],bioe:25,bleu:35,bleuevalu:35,bool:31,both:[16,26],bpe:29,bridg:[5,10,30],broadcast:41,bucket:23,budget:2,buffer:23,buffer_s:23,build:[33,39,41],build_cel:33,build_future_mask:41,build_sequence_mask:41,bureaucraci:2,cach:41,calcul:41,call:[19,35,38],callabl:[9,10,16,19,23,32,33,35,36,42],came:2,can:[1,2,10,19,23,28,31,43],capac:2,captur:31,case_insensitive_embed:21,categor:9,cell:[5,10,16,30],cell_class:[10,16,33],chang:34,charact:[1,21,29],charactertoken:29,charconvembedd:21,checkpoint:44,chief:44,chief_host:44,chunk:25,classifi:24,classnam:[23,38],client:31,cloth:2,code:[0,1],collabor:2,collect:[19,29,38],collection_nam:38,combin:40,combine_head:41,command:[1,44],common:[2,16],compat:32,complet:31,complex:1,comput:[25,31,37,41,44],compute_batch_indic:31,compute_topk_scores_and_seq:31,concaten:[40,41],concatreduc:[14,16,19,40],config:[1,4,5,23],config_path:6,configur:[2,3,6,9,20,21,24,25],consid:1,constant:[4,5,9,34],constrain:23,construct:10,constructor:16,contain:[1,19,20,21,23,24,25,26,29,31,40,41,43],content:2,context:41,contrib:[10,23,33],contribut:1,conv_encod:[5,12],convencod:13,conveni:33,converg:9,convert:2,convolut:[1,13,21],coodin:31,coordin:31,copi:36,copybridg:32,correctli:25,correspond:14,could:25,count_lin:38,count_paramet:38,counter:36,countershook:36,cpp:43,creat:[2,19,23,31,39,42],crf:25,crf_decod:25,cross:37,cross_entropy_loss:37,cross_entropy_sequence_loss:37,cuda_visible_devic:44,cumul:9,current:[0,19,25,26,27,31,34,36,44],custom:[2,36],data:[1,3,18,19,20,21,24,25,26,27,43,44],data_fil:[19,20,21],dataset:19,decai:[5,23,30],decay_r:[23,34],decay_step:[23,34],decay_typ:23,decayed_learning_r:23,decod:[4,5,26,31,32,38,39,44],decode_length:31,decoded_id:31,decoder_zero_st:32,defin:[1,2,6,7,8,10,11,12,13,15,16,17,18,19,20,21,22,27,28,29,30,32,34,37,39,40,41],definit:43,deliv:2,densebridg:32,depend:[19,38,39],depth:[2,20,39,40,41],describ:[1,2,9,11,13,16,17,23,27,34,41],design:1,detail:[21,23,44],dict:[26,31,38],dict_:38,dictionari:[6,19,23,26,29,38,41],differ:31,dimens:[16,21,27,34,40,41],dir:43,direct:16,directli:9,directori:[19,21,35,43,44],disk:[42,43],displai:[2,44],divis:16,doc:0,docker:44,document:[3,44],doe:32,done:19,dot:41,down:2,drop:[10,11,13,16,17,19,21,27,33,41],drop_and_add:41,dropout:[10,11,13,16,17,19,21,27,33,41],duplic:[1,2],dure:[9,39,43,44],dynam:[1,9,26,39],dynamic_decod:[9,10,11],dynamic_decode_and_search:[9,10,11],each:[9,10,14,16,20,21,25,27,31,33,37,38,39,40,41,42,44],easi:31,economi:2,ecosystem:44,element:[23,38,40],emb:39,embed:[1,9,10,11,18,19,21,44],embedd:21,embedding_fil:21,embedding_file_kei:21,embedding_file_with_head:21,embedding_s:21,embedding_var:21,empti:[6,35],encod:[1,4,5,10,24,25,26,32,38,39],encode_sequ:39,encoder_st:32,end:[1,2,9,26,31,36,43],end_token:[9,10,11],entri:[21,42],entropi:37,environ:43,eos_id:31,equal:31,estim:[9,14,19,23,33,37,41,44],etc:[1,25,44],european:2,evalu:[5,9,23,25,30,36,44],evaluators_nam:35,everi:14,every_n_sec:36,every_n_step:36,exampl:[1,2,19,23,43],execut:14,expand:31,expect:[2,20,43],expected_st:32,expos:38,extent:2,extern:[29,35,43],external_evaluation_fn:35,externalevalu:35,extract:24,extract_batch:38,extract_prefixed_kei:38,factor:16,fals:[10,16,19,21,25,33,34],fashion:23,feat1:2,featm:2,featur:23,features_fil:23,feder:2,feed:[11,17,27,41],feed_forward:41,ffn_inner_dim:[11,17,27],field:[19,20],file:[0,2,6,19,20,21,23,24,25,36,38,42,43],filenam:[29,36,38,42],fill:[6,42],filter:13,finish:31,first:[31,39,44],fit:2,flag:[25,31],flag_bioes_tag:25,flatten:20,float32:20,follow:[20,21],format:[21,23],forward:[11,17,19,27,41],found:[21,42],framework:44,freedom:1,frequenc:42,from:[0,1,2,9,10,11,13,17,18,20,21,23,24,27,31,32,36,38,41,42],full:[9,31],fund:2,futur:41,gather:[19,31],gener:[2,14,19,21,31,38],get:38,get_classnames_in_modul:38,get_dict_from_collect:38,get_embedding_fn:9,get_length:[19,20,21],get_optimizer_class:23,get_sampling_prob:9,get_serving_input_receiv:19,given:[19,26,31],global_step:[9,23,34],gnmt:16,goal:2,gold:25,gold_flag:25,googl:[27,41],googlernnencod:16,gpu:44,gradient:44,graph:[19,21,28,38,43,44],graphkei:29,greater:39,greedi:9,group:2,grow:31,grow_al:31,grow_finish:31,gym:2,hand:31,has:[1,2,20,25,31],have:[2,19,21,31,33,39],head:[2,11,17,27,41],help:2,helper:[33,40],here:31,hidden:[11,17,27,41],high:1,hook:[5,19,30,35],host:44,html:0,http:[9,11,13,16,17,27,31,34,41],ident:40,identifi:[2,25,42],identity_valu:40,ids:[9,26,31],ids_out:26,ignor:[21,25,34],implement:[1,2,18,31,36,41],implemet:31,impos:10,includ:[21,44],incompat:32,incorrectli:25,increas:[26,42],index:[2,31,39,42,43],infer:[1,9,10,11,43],inform:[32,43,44],inherit:[10,23],initi:[9,10,11,13,14,16,17,19,20,21,24,25,26,27,29,32,36,39,42],initial_id:31,initial_st:[9,10,11],inner:[10,11,16,17,27,33,41],inner_dim:41,input:[1,9,10,11,13,14,15,16,17,18,19,20,21,23,24,25,26,27,33,39,40,41,43],input_depth_kei:20,input_fn:23,inputt:[1,2,4,5,24,25,26,27],insensit:[21,24],inspect:43,inspir:31,instanc:[1,16,44],instead:9,instructor:2,int64:20,integ:[10,33],integr:44,interfac:31,introspect:31,invalid:[9,24,35],italian:2,item:31,item_or_tupl:38,iter:[9,21,38],its:[10,14,15,16,33,38,40,42],join:40,joinreduc:[14,40],kei:[1,2,19,20,21,24,25,26,38,41],kernel:13,kernel_s:[13,21],kill:2,know:43,known:[9,20],kubernet:44,label:[21,23,24,25,35,37],label_smooth:37,labels_fil:[23,35],labels_vocabulary_file_kei:[24,25],laid:2,last:[10,24],latest:43,layer:[10,11,13,16,17,25,27,33,41],learn:[1,23,34,44],learning_r:[23,34],learning_rate_decay_fn:23,legisl:2,length:[9,14,19,20,21,23,25,26,31,33,37,39,40,41],length_penalti:[9,10,11],level:1,like:[31,43,44],line:[1,2,18,21,24,25,38,42,44],linear:[9,11,17,32,41],list:[6,10,14,19,20,21,23,29,33,35,38,40,42,44],load:[6,29,42],load_config:6,load_model_modul:6,load_pretrained_embed:21,localhost:44,log:[9,19,21,31,35,36,44],log_dir:[19,21],log_prob:[9,31],logdir:44,logic:[18,32],logit:[9,10,31,37],logits_to_cum_log_prob:9,logparameterscounthook:36,look:43,lookup:[19,39,42],loop:44,loss:[5,30,44],lowercas:21,lstmcell:[10,16,33],luongattent:10,machin:44,made:[1,2],mai:31,main:[1,44],maintain:32,make:[0,31],make_dataset:[19,20,21],make_posit:39,manag:44,mani:[2,23],manual:43,mark:31,mask:41,master:2,match:[21,25],max:40,max_length:41,max_sequence_length:40,max_siz:42,max_word_length:21,maximum:[9,23,39,42],maximum_features_length:23,maximum_iter:[9,10,11],maximum_labels_length:23,maximum_posit:39,mean:[2,15,21],mean_encod:[5,12],meanencod:15,mechan:10,memori:[9,10,11,41],memory_sequence_length:[9,10,11],merg:[14,16,19,39,40],merge_config:1,meta:21,metadata:[19,20,21,23,29],method:[19,31],metric:25,min_frequ:42,minim:15,minimum:42,minimum_learning_r:23,misc:[5,30],miss:25,mix:[1,19],mixedinputt:[1,19],modal:1,mode:[9,10,11,13,14,15,16,17,19,20,21,23,33,37,41],modekei:[9,14,19,23,33,37,41],model:[2,4,5,6,19,29,31,34,36,41,43],model_fn:23,modul:[1,4,5,8,12,18,22,28,30],more:[1,23,35,44],mostli:36,multi:[1,10,11,14,17,19,33,35,41],multi_head_attent:41,multiattentionalrnndecod:10,multiinputt:19,multipl:[10,14,19],multipli:40,multiplyreduc:40,must:[9,21,31,32,40],my_config:1,name:[2,23,24,25,26,27,31,34,35,38],nation:2,need:[31,43],neg:25,neither:21,nest:[9,19,31,40],newlin:2,next:[10,23,31],noam_decai:34,node:43,non:6,none:[6,9,10,11,13,14,15,16,17,19,21,23,24,25,26,31,32,33,34,35,36,38,39,41,42],nor:21,norm:[41,44],normal:41,note:43,num_bucket:23,num_head:[11,17,27,41],num_lay:[10,11,13,16,17,27,33],num_oov_bucket:21,num_output:21,num_parallel_process_cal:23,num_unit:[10,11,13,16,17,27,33,41],number:[9,10,11,13,16,17,21,23,27,31,33,36,38,41,42],numpi:[21,25,38],object:[9,11,13,14,16,17,19,21,23,27,29,32,35,39,40,42],observ:31,occur:35,off:31,offset:40,onc:19,one:[10,19,21,24,25,35,39],onli:[0,10,25,26,27],open:44,opennmt:[0,1,2,3,43,44],oper:31,ops:[10,16,29,31,33],opt:23,optim:[1,21,23],option:[1,2,9,21,23,36],order:31,org:[9,11,13,16,17,27,31,34,41],other:[1,21,43],otherwis:[14,21,29],output:[9,10,11,13,14,16,17,21,23,24,26,27,31,33,35,36,38,41,43],output_dir:[35,36],output_fil:36,output_is_attent:10,outputs_reduc:14,over:13,overrid:10,packag:[0,3,4],pad:[19,31,39,40],pad_in_tim:40,pad_n_with_ident:40,pad_with_ident:40,padded_batch:19,padded_shap:19,padding_length:40,pair:19,parallel:[19,23],parallelencod:[1,14],parallelinputt:[1,2,19],param:[23,24,25,26],paramet:[6,9,10,11,13,14,16,17,19,20,21,23,24,25,26,27,29,31,32,33,34,35,36,37,38,39,40,41,42,44],parameter:32,parliament:2,part:1,pass:[10,32],past:23,path:[6,21,35,42],pattern:31,pbtxt:21,penal:31,penalti:[9,31],per:[21,24,25],period:43,perl:35,permit:31,person:2,pilat:2,pip:0,portal:3,pos:31,posit:[5,11,13,17,25,27,30,41],position_encod:[11,13,17,27],positionembedd:[11,13,17,27,39],positionencod:[11,13,17,27,39],possibl:[6,9,19,31,40],post_evaluation_fn:36,postur:2,pre:41,precis:25,predefin:1,predict:[23,24,25,26,35,36],predicted_flag:25,predicted_id:9,predictions_path:35,prefer:1,prefetch:23,prefix:[2,26,31,38],prepar:[2,19,26],prerog:2,pretrain:21,previou:41,print:[23,38],print_byt:38,print_predict:[23,24,25,26],prioriti:1,probabl:[9,10,11,13,16,17,19,21,27,31,33,37,41,44],process:[19,21,23,24,25,26,27,31],produc:25,product:41,program:2,project:[7,38,41],projector_config:21,provid:[1,3,21,31,43],provis:2,prune:42,ps_host:44,purchas:2,purpos:2,py_func:43,pyramidalrnnencod:16,python:[0,1,2,6,10,16,29,33,36,42,43,44],queri:[9,41],quot:2,rais:[9,16,21,23,24,26,32,33,35],rate:[23,34,44],raw:[18,19],reach:31,read:[9,20,38],read_prob:9,realis:2,recal:25,receiv:19,recommonmark:0,record:[2,20,31],record_inputt:[5,18],reduc:[5,14,16,19,30,37,39],reduce_sequ:40,reduced_input:40,reduced_length:40,reduct:16,reduction_factor:16,refer:32,reflect:26,refus:2,regist:[21,29],rel:[6,21],relat:[1,6,35,41],relu:[11,17,27],relu_dropout:[11,17,27],remov:[0,19],remove_data_field:19,renam:0,replac:10,replic:41,repres:31,requir:[19,31,43],reserv:39,residu:41,residual_connect:[10,16,33],resolv:23,respect:40,result:[21,35],retain:40,reusabl:1,rightmost:1,rip:2,rnn:[1,10,16,33],rnn_cell_impl:[10,16,33],rnn_decod:[5,8],rnn_encod:[5,12],rnncell:33,rnndecod:10,rnnencod:16,roll_sequ:40,run:[0,31,35,43,44],run_context:36,run_valu:36,sai:[2,31],same:[14,19,33,39,40],sampl:[1,9,44],sampling_prob:[9,10,11],save:[16,35,36,42],saved_model:43,saved_model_cli:43,saveevaluationpredictionhook:[35,36],scale:[2,34,41],scaled_dot_attent:41,schedul:9,schedule_typ:9,scheme:25,scor:31,score:[31,35,41],scores_to_gath:31,script:[1,2,42,43],seach:31,search:[9,31],see:[1,2,19,44],select:44,self:[1,11,17,27],self_attention_decod:[5,8],self_attention_encod:[5,12],selfattentiondecod:11,selfattentionencod:17,semant:34,senior:2,sentenc:[2,31,42],separ:[1,2],seq2seq:[10,26,33],seq:31,seq_length:31,seqclassifi:24,seqtagg:25,sequenc:[9,13,14,16,18,21,23,24,25,26,27,31,37,39,40,41],sequence_classifi:[5,22],sequence_length:[9,10,11,13,14,15,16,17,21,25,37,39,40,41],sequence_tagg:[5,22],sequence_to_sequ:[5,22,27],sequenceclassifi:24,sequencerecordinputt:[2,20],sequencetagg:25,sequencetosequ:[26,27],sequenti:[14,31],sequentialencod:[1,14],serial:[2,42,43],serv:[3,19,23],server:44,serving_default:43,serving_input_fn:23,servinginputreceiv:[19,23],session:36,session_run_hook:36,sessionrunhook:36,set:[1,2,9,19,21,23,29,32,38,39,41,44],set_data_field:19,setup:19,sever:[14,19,44],shape:[9,14,19,20,21,39,40,41],share:27,shell:44,shift:[26,40],shift_target_sequ:26,shortcut:39,shoud:31,should:[1,2,26,29,44],show:43,shuffl:23,signatur:[19,23],signature_def:43,similar:23,simpl:[10,15,16,19,21,36],simpli:[1,44],simplifi:2,singl:[1,38,40],site:2,size:[9,13,21,23,31,32,42],sleep:2,smooth:37,soldier:2,some:[1,43],sourc:[1,2,6,9,10,11,13,14,15,16,17,19,20,21,23,24,25,26,27,29,31,32,33,34,35,36,37,38,39,40,41,42,44],source_inputt:[26,27],space:[2,21,29],spacetoken:[21,29],special:42,special_token:42,specif:19,speed:44,sphinx:0,sphinx_rtd_them:0,split:[21,29,41],split_head:41,src:[2,43],staircas:[23,34],standard:[26,29],start:[9,21,23,26,31,42,44],start_decay_step:23,start_token:[9,10,11],state:[2,9,10,14,16,31,32],states_reduc:14,states_to_gath:31,stdout:38,step:[9,10,23,31,34,35,36],stepcounterhook:36,stop:26,store:[23,31],str_as_byt:38,stream:[23,24,25,26,38],stretch:2,stride:21,string:[23,25,28,29,31,38,42],structur:[19,40],submodul:4,subpackag:4,suffix:[26,36],sum:40,summar:36,summari:44,summary_writ:36,sumreduc:[16,39,40],support:[25,26,27,43,44],sybmol:31,symbol:31,symbols_to_logits_fn:31,synchron:44,synergi:2,sys:38,tabl:[19,39],table_initi:19,tag:25,tag_set:43,tagger:25,tagging_schem:25,take:[9,10,15,16,31,33,36],target:[2,9,26,27],target_inputt:[26,27],task:44,task_index:44,task_typ:44,templat:1,tensor:[9,19,20,21,26,28,29,31,38,39,40,41],tensorboard:[21,44],tensorflow:[10,16,33,36,43,44],term:2,test:0,text:[18,21,29,42,43],text_inputt:[5,18,26,27],textinputt:[2,21],tfdbg:31,tfrecord:[2,20],tgt:43,than:39,thei:[1,2],them:2,thi:[2,3,10,19,21,23,24,25,26,27,31,35,36,38,39,43],thing:31,thoughout:7,three:31,throughout:38,tile:41,tile_sequence_length:41,time:[16,20,31,40,41],titl:2,toi:[1,2,43],token:[2,4,5,9,21,26,42,43],tokens_to_char:21,tokp_gathered_scor:31,told:2,top:31,topk:31,topk_finished_flag:31,topk_gathered_scor:31,topk_seq:31,total:38,train:[1,2,3,9,10,11,13,14,15,16,17,19,21,23,36,37,43],train_and_evalu:44,train_features_fil:2,train_source_1:2,train_source_2:2,train_source_3:2,trainabl:[21,36,38],trainer:2,transform:[5,11,17,18,19,20,21,22,28,30,32],transform_data:19,tupl:[9,14,25,31,38,40,41],two:2,txt:[2,43],type:[9,23,31,43],typeerror:[9,26],typic:43,unicod:[21,29],unidirectionalrnnencod:16,uniqu:31,unit:[10,11,13,16,17,19,21,27,33,41],unknown:21,unless:2,unscal:37,unsur:1,unused_data:19,updat:[19,26],url:44,usag:2,use:[0,23,31,38,41,43],used:[1,2,7,9,16,23,25,31,35,39,43],user:[1,19,23,29,44],uses:44,using:[1,2,10,11,16,17,31,43],usual:[9,19,39],util:[4,5,10,11,13,14,16,17,19,23,27],val1:21,val2:21,valm:21,valu:[1,7,9,19,20,23,31,37,38,39,40,41,42],valueerror:[9,16,21,23,24,32,33,35],variabl:[20,21,43],variant:21,variou:[30,38],vector:[18,21,24,41],version:42,view:38,visual:[19,21,31,44],visualize_embed:21,vocab:[5,30,31,43],vocab_s:[9,10,11,21,31],vocabulari:[9,21,24,25,42,43],vocabulary_fil:21,vocabulary_file_kei:21,vocabulary_s:21,volatil:19,warmup:34,watch:31,weight:[9,43],welcom:1,when:[2,9,16,23,25,31,43],where:[2,9,39,40,41,42],whether:31,which:[2,10,19,20,31,33,35,36,40],whose:2,width:9,window:21,with_head:21,within:19,word1:21,word2:21,word:[1,9,21,44],wordembedd:[21,26,27],wordn:21,work:[1,28],worker:44,worker_host:44,wrapper:10,write:42,yaml:[1,2],yml:1,you:[1,2,43],your:[2,43],zerobridg:[10,32],zip:40,zip_and_reduc:40},titles:["Documentation","Configuration","Data","Overview","opennmt","opennmt package","opennmt.config module","opennmt.constants module","opennmt.decoders package","opennmt.decoders.decoder module","opennmt.decoders.rnn_decoder module","opennmt.decoders.self_attention_decoder module","opennmt.encoders package","opennmt.encoders.conv_encoder module","opennmt.encoders.encoder module","opennmt.encoders.mean_encoder module","opennmt.encoders.rnn_encoder module","opennmt.encoders.self_attention_encoder module","opennmt.inputters package","opennmt.inputters.inputter module","opennmt.inputters.record_inputter module","opennmt.inputters.text_inputter module","opennmt.models package","opennmt.models.model module","opennmt.models.sequence_classifier module","opennmt.models.sequence_tagger module","opennmt.models.sequence_to_sequence module","opennmt.models.transformer module","opennmt.tokenizers package","opennmt.tokenizers.tokenizer module","opennmt.utils package","opennmt.utils.beam_search module","opennmt.utils.bridge module","opennmt.utils.cell module","opennmt.utils.decay module","opennmt.utils.evaluator module","opennmt.utils.hooks module","opennmt.utils.losses module","opennmt.utils.misc module","opennmt.utils.position module","opennmt.utils.reducer module","opennmt.utils.transformer module","opennmt.utils.vocab module","Serving","Training"],titleterms:{autodoc:0,beam_search:31,bridg:32,build:0,cell:33,config:6,configur:1,constant:7,conv_encod:13,data:2,decai:34,decod:[8,9,10,11],depend:0,distribut:44,document:0,encod:[12,13,14,15,16,17],evalu:35,file:1,format:2,hook:36,input:2,inputt:[18,19,20,21],instal:0,local:0,loss:37,mean_encod:15,misc:38,model:[1,22,23,24,25,26,27],modul:[6,7,9,10,11,13,14,15,16,17,19,20,21,23,24,25,26,27,29,31,32,33,34,35,36,37,38,39,40,41,42],monitor:44,multipl:1,opennmt:[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42],overview:3,packag:[5,8,12,18,22,28,30],parallel:2,paramet:1,posit:39,record_inputt:20,reduc:40,regist:0,rnn_decod:10,rnn_encod:16,self_attention_decod:11,self_attention_encod:17,sequence_classifi:24,sequence_tagg:25,sequence_to_sequ:26,serv:43,sourc:0,submodul:[5,8,12,18,22,28,30],subpackag:5,text:2,text_inputt:21,token:[28,29],train:44,transform:[27,41],util:[30,31,32,33,34,35,36,37,38,39,40,41,42],vector:2,vocab:42}})