/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package capstone_demo;

import java.io.*;
/**
 *
 * @author Matt Q
 */
public class ProcessBuilderDemo {
    public static void main(String[] args) throws Exception{
        
      // create a new list of arguments for our process
      String[] list = {"java", "-jar", "AuToBI.jar", ""};

      // create the process builder
      ProcessBuilder pb = new ProcessBuilder(list);
      File autobi_directory = new File("C:/Users/Matt Q/eclipse-workspace/AuToBI");

      // set the command list
      pb.command(list);
      pb.directory(autobi_directory);
      pb.inheritIO();

      // print the new command list
      System.out.println("" + pb.command());
      
      Process p = pb.start();
      
    }
    
}
