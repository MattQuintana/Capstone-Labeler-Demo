package capstone_demo;

public class numJava {

	// Transpose an array 
	public double[][] transpose(double[][] array)
	{
		double[][] result = new double[array[0].length][array.length];
		
		// For every column in the source
		for (int column = 0; column < array[0].length; column++)
		{
			// For every entry in the column
			for (int row = 0; row < array.length; row++)
			{
				// Fill in the transposed array
				result[column][row] = array[row][column];
			}
		}
		
		// Return the newly filled array
		return result;
	}
	
	public double[][] transpose1D(double[] array)
	{
		double[][] result = new double[array.length][1];
		
		for (int i = 0; i < array.length; i++)
		{
			result[i][0] = array[i];
		}
		return result;
	}
	
	// Normalize a vector to have values that add to one
	public double[] normalizeVector(double[] array)
	{
		double sum = 0;
		// Get the sum of the values in the vector
		for(int i = 0; i < array.length; i++) 
        {
        	array[i] = this.nonlin(array[i], false);
        	sum += array[i];
        }
        
        // Sum the array to 1 
        for (int i = 0; i < array.length; i++)
        {
        	array[i] /= sum;
        }
		return array;
	}
	
	// Normalize a value to be between 0 and 1
	public double nonlin(double x, boolean deriv) 
	{
		if (deriv == true)
		{
			return x*(x-1);
		}
    	return 1/(1 + Math.exp(-x));
    }
	
	// Call the nonlin function on a vector to change every entry
	public double[] nonlinVector(double[] x, boolean deriv)
	{
		double[] result = new double[x.length];
		for (int i = 0; i < x.length; i++)
		{
			result[i] = nonlin(x[i], deriv);
		}
		
		return result;
	}
	
	// Add two arrays together and get the result
	public double[] add(double[] a, double[] b)
	{
		if (a.length != b.length)
		{
			throw new java.lang.Error("Mismatch in array size when adding.");
		}
		double[] result = new double[a.length];
		for (int i = 0; i < a.length; i++)
		{
			
		}
		
		return result;
	}
	
	public double[][] add2D(double[][] a, double[][] b)
	{
		double[][] result = new double[a.length][a[0].length];
		
		for (int i = 0; i < a.length; i ++)
		{
			for (int j = 0; j < a[0].length; j++)
			{
				result[i][j] = a[i][j] + b[i][j];
			}
		}
		
		return result;
	}
	
    // return c = a - b
      public double[] subtract(double[] a, double[] b) {
        int m = a.length;
        double[] result = new double[m];
        for (int i = 0; i < m; i++)
                result[i] = a[i] - b[i];
        return result;
    }
    
    // return c = a * b
    public double[] multiply(double[] a, double[] b) {
        int a_len = a.length;
        int b_len = b.length;
        if (a_len != b_len) throw new RuntimeException("Illegal matrix dimensions.");
        double[] c = new double[a_len];
        for (int i = 0; i < a_len; i++) 
        {
        	c[i] = a[i] * b[i];
        }
        return c;
    }
	
	// Perform the dot multiplication of a 1D vector against a 2D matrix
	public double[] dot1D(double[] vector, double[][] matrix)
	{
		// Prepare a result array
		double[] result = new double[matrix[0].length];
		
		// Do the matrix multiplication
    	// Basically multiply all of the mean values retrieved from the attributes 
    	// by the randomized weights in the matrix. 
        for(int matx_col = 0; matx_col < matrix[0].length; matx_col++) {
        	double col_sum = 0;
        	for(int matx_row = 0; matx_row < matrix.length; matx_row++) {
        		col_sum += vector[matx_row] * matrix[matx_row][matx_col];
        	}
        	// Put all of those multiplications into the corresponding output layer node
        	result[matx_col] = col_sum;
        }
		
		return result;
	}
	
	// Get the dot product of two 2d matrices 
	public double[][] dot2D(double[][] array_a, double[][] array_b)
	{
		double[][] result = new double[array_a.length][array_b[0].length];
		
		// Catch the error if the arrays are not correct in size
		if (array_a[0].length != array_b.length)
		{
			System.out.println(String.format("Array mismatch of size:%d X %d vs %d X %d", array_a.length, array_a[0].length, array_b.length, array_b[0].length));
			throw new java.lang.Error("Size mismatch in arrays.");
		}
		
		// For every row in the first matrix
		for (int row = 0; row < result.length; row++)
		{
			// For every column in the second matrix
			for (int col = 0; col < result[0].length; col++)
			{
				// Prepare a result to be stored
				double sing_product = 0;
				
				// Multiply the entries in each row and column and get sum
				for (int entry = 0; entry < result[0].length; entry++)
				{
					sing_product += array_a[row][entry] * array_b[entry][col]; 
				}
				
				result[row][col] = sing_product;
			}
		}
		return result;
	}
	
	
	public double[][] dot1D_rev(double[][] matrix, double[] vector)
	{
		
		if (matrix[0].length != 1)
		{
			throw new java.lang.Error("Mismatch in matrix and vector dimensions.");
		}
		
		double[][] result = new double[matrix.length][vector.length];
		
		// For every row in the matrix
		for (int i = 0; i < matrix.length; i++)
		{
			// For every entry in the vector 
			for (int j = 0; j < vector.length; j++)
			{
				// Multiply and put in resulting matrix
				result[i][j] = matrix[i][0] * vector[j];
			}
		}
		
		return result;
	}
}
