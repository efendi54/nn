#pragma once
#include "net/layer.h"

/**
 * \brief Structure to store feed-forward-nets related data
 * */
struct FFNet
{
  /**
   * \brief Constructor that creates a net with appropriate architecture
   * \param p_Arch vector of layer-sizes (e.g. a vector with thre elements {2,3,3}
   *        would create a net of three layers with neuron amounts of two in the 
   *        input layer, and three neurons in the middle- and output-layer.
   * */
  FFNet(const std::vector<size_t> &p_Arch);

  /**
   * \brief Prints net-state to std-out
   * */
  void Print(void);

  /**
   * \brief Feeding in one data-item to be learned
   * \param p_X data item of length n representing n-features
   * */
  void Feed(const DVec &p_X);

  /**
   * \brief Backpropagation of net errors in regard to provided target-output
   * \param p_Y target outputs of the net with whom errors should be backpropagated
   * */
  void BackProp(const DVec &p_Y);

  /**
   * \brief Learning algorithm representing the backpropagation learn-algorithm for
   *        feed-forward nets
   * \param p_X vector of data-items to be learned
   * \param p_Y vector of target-items to be learned
   * \param p_Iterations number of (fixed) iterations to loop through whole
   *        provided data-items
   * */
  void Learn(const std::vector<DVec> &p_X,
             const std::vector<DVec> &p_Y,
             const ulong p_Iterations=1000);

  double t_LearnRate;           ///< learning rate of the net
  double t_Momentum;            ///< momentum used during learning
  std::vector<Layer> t_Layers;  ///< all layers of the net
};

