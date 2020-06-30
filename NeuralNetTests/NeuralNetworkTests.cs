using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNet;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.Tests
{
    [TestClass()]
    public class NeuralNetworkTests
    {
        [TestMethod()]
        public void FeedForwardTest()
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

            var topology = new Topology(4, 1, 0.001, 10);
            NeuralNetwork neuralNetwork = new NeuralNetwork(topology);
            var diffrence = neuralNetwork.Learn(dataset, 100000);
            var results = new List<double>();

            foreach(var data in dataset)
            {
                results.Add(neuralNetwork.FeedForward(data.Item2).Neurons[0].Output);
            }

            for (int i = 0; i< results.Count; i++)
            {
                var expected = Math.Round(dataset[i].Item1[0], 4);
                var actual = Math.Round(results[i], 4);
                Assert.AreEqual(expected, actual);
            }
        }
    }
}