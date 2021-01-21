package org.bert;

import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.base.Stopwatch;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.RawTensor;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.*;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.types.*;
import java.util.Map;
import org.tensorflow.types.TFloat32;
import java.io.File;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.lang.Math;
import org.apache.commons.cli.*;

public class BertSentenceSimilarity {

  public static void main(String[] args) throws Exception {
    
    final String SEPARATOR_TOKEN = "[SEP]";
    final String START_TOKEN = "[CLS]";
    CommandLine cmd = parse_args(args);
    final String VOCAB_FILE = cmd.getOptionValue("vocab");//"/Users/suneetsawant/Downloads/bert_en_uncased_L-12_H-768_A-12_3/assets/vocab.txt";
    String modelPath = cmd.getOptionValue("model");//"/Users/suneetsawant/Downloads/bert_model";
    String sent1 = cmd.getOptionValue("sentence1");//"This is Hello World.A great program";
    String sent2 = cmd.getOptionValue("sentence2");//"this is random bullshit";

    System.out.println("Sentence1: "+sent1);
    System.out.println("Sentence2: "+sent2);
    System.out.println("Model Path:"+modelPath);
    System.out.println("Vocab Path"+VOCAB_FILE);

    final int separatorTokenId;
    final int startTokenId;
    final FullTokenizer tokenizer;

    Path vocabulary = Paths.get(VOCAB_FILE);
    boolean dolowercase = true;

    tokenizer = new FullTokenizer(vocabulary, dolowercase);

    final int[] ids = tokenizer.convert(new String[] {START_TOKEN, SEPARATOR_TOKEN});
    startTokenId = ids[0];
    separatorTokenId = ids[1];

    
    Inputs inputs = getInputs(sent1, sent2,tokenizer,startTokenId,separatorTokenId);
    
    try (SavedModelBundle model = SavedModelBundle
        .load(modelPath, "serve")) {
      Session sess = model.session();
      TFloat32 out = (TFloat32) sess
          .runner()
          .feed("serving_default_input_word_ids", inputs.inputIds)
          .feed("serving_default_input_mask", inputs.inputMask)
          .feed("serving_default_input_type_ids", inputs.inputTypeIds)
          .fetch("StatefulPartitionedCall")
          .run().get(0);
      
      double score = out.getFloat(0,0);
  
      System.out.println("Sentence1: "+sent1);
      System.out.println("Sentence2: "+sent2);
      System.out.println("Similarity Score (0-1): " + score);
    }
  }

  private static CommandLine parse_args(String[]args){
    Options options = new Options();

    Option model_path = new Option("m", "model", true, "path to trained model");
    model_path.setRequired(true);
    options.addOption(model_path);

    Option vocab = new Option("v", "vocab", true, "BertVocabPath-en_uncased");
    vocab.setRequired(true);
    options.addOption(vocab);

    Option sent1 = new Option("s1", "sentence1", true, "Input Sentence1");
    sent1.setRequired(true);
    options.addOption(sent1);

    Option sent2 = new Option("s2", "sentence2", true, "Input Sentence2");
    sent2.setRequired(true);
    options.addOption(sent2);

    CommandLineParser parser = new DefaultParser();
    HelpFormatter formatter = new HelpFormatter();
    CommandLine cmd=null;

    try {
        cmd = parser.parse(options, args);
    } catch (ParseException e) {
        System.out.println(e.getMessage());
        formatter.printHelp("utility-name", options);
        System.exit(1);
    }

    return cmd;
  }
  
  private static Tensor fromArrayToTensor(int [] input){
    FloatNdArray input_matrix = NdArrays.ofFloats(Shape.of(1,input.length));

    for (int i = 0; i < input.length; i++) {
      input_matrix.setFloat((float)input[i],0,i);
    }
    
    Tensor tensor = TFloat32.tensorOf(input_matrix);
    return tensor;
  }

  private static Inputs getInputs(final String sent1,final String sent2,FullTokenizer tokenizer,final int startTokenId,final int separatorTokenId) {
    final String[] tokens1 = tokenizer.tokenize(sent1);
    final String[] tokens2 = tokenizer.tokenize(sent2);
    final int maxSequenceLength = 128;
    final IntBuffer inputIds = IntBuffer.allocate(maxSequenceLength);
    final IntBuffer inputMask = IntBuffer.allocate(maxSequenceLength);
    final IntBuffer inputTypeIds = IntBuffer.allocate(maxSequenceLength);

    int[] ids = tokenizer.convert(tokens1);
    inputIds.put(startTokenId);
    inputMask.put(1);
    inputTypeIds.put(0);
    for(int i = 0; i < ids.length; i++) {
        inputIds.put(ids[i]);
        inputMask.put(1);
        inputTypeIds.put(0);
    }
    inputIds.put(separatorTokenId);
    inputMask.put(1);
    inputTypeIds.put(0);

    ids = tokenizer.convert(tokens2);
    for(int i = 0; i < ids.length && i < maxSequenceLength - 2; i++) {
      inputIds.put(ids[i]);
      inputMask.put(1);
      inputTypeIds.put(1);
  }
  inputIds.put(separatorTokenId);
  inputMask.put(1);
  inputTypeIds.put(1);

    inputIds.rewind();
    inputMask.rewind();
    inputTypeIds.rewind();
    return new Inputs(inputIds.array(), inputMask.array(), inputTypeIds.array());
}
private static class Inputs implements AutoCloseable {
  private final Tensor inputIds, inputMask, inputTypeIds;

  public Inputs(final int[] inputIds, final int[] inputMask, final int[] inputTypeIds) {
      this.inputIds = fromArrayToTensor(inputIds);
      this.inputMask = fromArrayToTensor(inputMask);
      this.inputTypeIds = fromArrayToTensor(inputTypeIds);
  }

  @Override
  public void close() {
      inputIds.close();
      inputMask.close();
      inputTypeIds.close();
  }
}

}
