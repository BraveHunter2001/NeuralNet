using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    class Program
    {
        static void Main(string[] args)
        {
            // самый ебаннутый дата сет 
            var dataset = new List<Tuple<double[], double[]>>
            {
                // Результат - Пациент болен - 1
                //             Пациент Здоров - 0

                // Неправильная температура T
                // Хороший возраст A
                // Курит S
                // Правильно питается F
                //                                             T  A  S  F
                new Tuple<double[], double[]> (new double[] { 0 }, new double[] { 0, 0, 0, 0 }),
                new Tuple<double[], double[]> (new double[] { 0 }, new double[] { 0, 0, 0, 1 }),
                new Tuple<double[], double[]> (new double[] { 1 }, new double[] { 0, 0, 1, 0 }),
                new Tuple<double[], double[]> (new double[] { 0 }, new double[] { 0, 0, 1, 1 }),
                new Tuple<double[], double[]> (new double[] { 0 }, new double[] { 0, 1, 0, 0 }),
                new Tuple<double[], double[]> (new double[] { 0 }, new double[] { 0, 1, 0, 1 }),
                new Tuple<double[], double[]> (new double[] { 1 }, new double[] { 0, 1, 1, 0 }),
                new Tuple<double[], double[]> (new double[] { 0 }, new double[] { 0, 1, 1, 1 }),
                new Tuple<double[], double[]> (new double[] { 1 }, new double[] { 1, 0, 0, 0 }),
                new Tuple<double[], double[]> (new double[] { 1 }, new double[] { 1, 0, 0, 1 }),
                new Tuple<double[], double[]> (new double[] { 1 }, new double[] { 1, 0, 1, 0 }),
                new Tuple<double[], double[]> (new double[] { 1 }, new double[] { 1, 0, 1, 1 }),
                new Tuple<double[], double[]> (new double[] { 1 }, new double[] { 1, 1, 0, 0 }),
                new Tuple<double[], double[]> (new double[] { 0 }, new double[] { 1, 1, 0, 1 }),
                new Tuple<double[], double[]> (new double[] { 1 }, new double[] { 1, 1, 1, 0 }),
                new Tuple<double[], double[]> (new double[] { 1 }, new double[] { 1, 1, 1, 1 })
            };


            int OutputNeuron = 1; // кол-во выходных нейронов
            int inputNeuron = 4;  // кол-во входных нейронов
            double learningRate = 0.01; // скорость обучения
            int epoch = 50000; // кол-во эпох обучения

            Topology topology = new Topology(inputNeuron, OutputNeuron,learningRate, 4); // после learningRate идет инициализация скрытых словев с количесвом нейронов через запятую.
            // в данном случае стоит одна 4 => создан один скрытый слой с 4 нейронами 

            NeuralNetwork neuralNetwork = new NeuralNetwork(topology);

            double diffrence = neuralNetwork.Learn(dataset, epoch);

            double[,] results = new double[dataset.Count, OutputNeuron];

            int k = 0;
            foreach (var data in dataset)
            {
        
                for (int i =0; i < OutputNeuron; i++)
                    results[k, i] = neuralNetwork.FeedForward(data.Item2).Neurons[i].Output;
                k++;
            }

            Console.WriteLine("Ошибка в обучении [{0}]: ", diffrence);
            for (int i =0; i < dataset.Count; i++)
            {
                Console.Write("Результат:" + i + "|");
                for (k =0; k < OutputNeuron; k++)
                    Console.Write("Neuron[{0}]: {1}|", i, Math.Round(results[i, k],4));
                Console.WriteLine();    
            }
            Console.ReadKey();
        }
    }
}
