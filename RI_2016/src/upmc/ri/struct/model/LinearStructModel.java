package upmc.ri.struct.model;

import upmc.ri.struct.instantiation.IStructInstantiation;

public abstract class LinearStructModel<X, Y> implements IStructModel<X, Y> {
	IStructInstantiation<X, Y> Inst;
	//what about this instantiation method ?
	double[] w;
	
	public IStructInstantiation<X, Y> instantiation(){
		assert(Inst!=null);
		return this.Inst;
	}
	
	public double[] getParameters() {
		return w;
	}
	
	public LinearStructModel(int dim_params,IStructInstantiation<X, Y> Inst ) {
		this.w = new double[dim_params * Inst.enumerateY().size()];
		this.Inst = Inst;
	}
	
	
	
}
