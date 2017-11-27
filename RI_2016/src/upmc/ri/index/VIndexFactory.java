package upmc.ri.index;
import upmc.ri.index.*;
import java.util.Set;
import java.util.stream.DoubleStream;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class VIndexFactory {
	public static double[] computeBow(ImageFeatures ft){
		List<Integer> words = ft.getwords();
		int vocab_size = ImageFeatures.tdico;
		
		//create bow (hard assignment / sum pooling)
		double[] bow = new double[vocab_size];
		for (int w : words) {
			bow[w]++;
		}
		
		//l2 norm
		double sum = DoubleStream.of(bow).sum();
		bow = Arrays.stream(bow).map(k -> k / sum).toArray();
		return bow;
		
	}
}
