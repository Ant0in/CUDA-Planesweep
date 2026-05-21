#include "../kernels/kernel.cuh"
#include "cam_params.hpp"
#include "constants.hpp"
#include "graph.h"

#include <cstdio>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <string>
#include <climits>  // for SHRT_MAX

/**
 * @brief Loads all camera images and attaches the hardcoded calibration parameters.
 * @param folder Folder containing v0.png, v1.png, and friends.
 * @return Cameras ready for the CPU/CUDA planesweep pipeline.
 */
std::vector<cam> read_cams(std::string const &folder) {

	// init parameters and cameras
	std::vector<params<double>> cam_params_vector = get_cam_params();
	std::vector<cam> cam_array(cam_params_vector.size());

	for (int i = 0; i < cam_params_vector.size(); i++) {

		std::string name = folder + "/v" + std::to_string(i) + ".png";  // cam format is v*.png
		cv::Mat im_rgb = cv::imread(name);
		cv::Mat im_yuv;
		const int width = im_rgb.cols;
		const int height = im_rgb.rows;

		// convert to YUV420 because that's what the kernel uses (and because cpu implementation was doing it)
		cv::cvtColor(im_rgb, im_yuv, cv::COLOR_BGR2YUV_I420);
		const int size = width * height * 1.5; // YUV 420

		std::vector<cv::Mat> YUV;
		cv::split(im_rgb, YUV);

		// params
		cam_array.at(i) = cam(name, width, height, size, YUV, cam_params_vector.at(i));
	}

	return cam_array;

	// ?? that was used for testing i'll keep it there just in case
	// cv::Mat U(height / 2, width / 2, CV_8UC1, cam_array.at(0).image.data() + (int)(width * height * 1.25));
	// cv::namedWindow("im", cv::WINDOW_NORMAL);
	// cv::imshow("im", U);
	// cv::waitKey(0);

}

/* !! The next two function are used to perform the graph cut on the results
DO NOT MODIFY THOSE FUNCTIONS - DO NOT TRY TO IMPLEMENT THEM ON THE GPU !!*/
// ^^^^^^ ok

/**
 * @brief Helper used by the graph-cut refinement to connect neighboring pixels.
 * @param g Graph object being built for the current depth layer.
 * @param nodes Per-pixel graph nodes.
 * @param destPixel Neighbor pixel we connect to.
 * @param sourcePixel Current pixel.
 * @param imgSize Image dimensions.
 * @param m_aiEdgeCost Smoothness costs per label difference.
 * @param labels Current depth labels.
 * @param label Candidate label being tested.
 * @param cost_cur Smoothness cost for the current edge.
 */
void depth_estimation_by_graph_cut_sWeight_add_nodes(Graph& g, std::vector<Graph::node_id>& nodes, cv::Size destPixel, cv::Size sourcePixel, cv::Size imgSize, std::vector<double> m_aiEdgeCost, cv::Mat1w labels, int label, double cost_cur) {
	const int idxSourcePixel = sourcePixel.height * imgSize.width + sourcePixel.width;
	const int idxDestPixel = destPixel.height * imgSize.width + destPixel.width;
	const double cost_cur_temp = cost_cur;

	if (labels(sourcePixel.height, sourcePixel.width) != labels(destPixel.height, destPixel.width)) {
		//add a new node and add edge between it and the adjacent nodes
		Graph::node_id tmp_node = g.add_node();
		const double cost_temp = m_aiEdgeCost[std::abs(labels(destPixel.height, destPixel.width) - label)];
		g.set_tweights(tmp_node, 0, m_aiEdgeCost[std::abs(labels(sourcePixel.height, sourcePixel.width) - labels(destPixel.height, destPixel.width))]);
		g.add_edge(nodes[idxSourcePixel], tmp_node, cost_cur_temp, cost_cur_temp);
		g.add_edge(tmp_node, nodes[idxDestPixel], cost_temp, cost_temp);
	}
	else //only add an edge between two nodes
		g.add_edge(nodes[idxSourcePixel], nodes[idxDestPixel], cost_cur_temp, cost_cur_temp);
}

/**
 * @brief Refines the raw planesweep costs with graph cut, slow but prettier.
 * @param cost_cube Vector of per-depth cost images coming from the sweep.
 * @return 8-bit depth map after alpha-expansion-ish graph-cut refinement.
 */
cv::Mat depth_estimation_by_graph_cut_sWeight(std::vector<cv::Mat> const& cost_cube) {
	//DO NOT TRY TO IMPLEMENT THIS FUNCTION ON THE GPU

	printf("[-] CUDA graph-cut extraction\n");

	const int zPlanes = cost_cube.size();
	const int height = cost_cube[0].size().height;
	const int width = cost_cube[0].size().width;

	//To store the depth values assigned to each pixels, start with 0
	cv::Mat1w labels = cv::Mat::zeros(height, width, CV_16U); 
	//store the cost for a label
	std::vector<double> m_aiEdgeCost;
	double smoothing_lambda = 1.0;
	m_aiEdgeCost.resize(zPlanes);
	for (int i = 0; i < zPlanes; ++i)
		m_aiEdgeCost[i] = smoothing_lambda * i;

	for (int source = 0; source < zPlanes; ++source) {
		std::cout << EA_GRAY << "[-] Depth Layer " << source << "/" << zPlanes - 1 << EA_DEFAULT << std::flush;
		Graph g;
		std::vector<Graph::node_id> nodes(height * width, nullptr);

		//Putting the weights for the connection to the source and the sink for each nodes
		for (int r = 0; r < height; ++r) {
			for (int c = 0; c < width; ++c) {
				//indice global du pixel
				const int pp = r * width + c;
				nodes[pp] = g.add_node();
				const ushort label = labels(r, c);
				if (label == source)
					g.set_tweights(nodes[pp], cost_cube[source].at<float>(r, c), SHRT_MAX);
				else
					g.set_tweights(nodes[pp], cost_cube[source].at<float>(r, c), cost_cube[label].at<float>(r, c));
			}
		}

		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				const double cost_curr = m_aiEdgeCost[std::abs(labels(j, i) - source)];

				//create an edge between the adjacent nodes, may add an additional node on this edge if the previously calculated labels are different
				if (i != width - 1) {
					depth_estimation_by_graph_cut_sWeight_add_nodes(g, nodes, cv::Size(i + 1, j), cv::Size(i, j), cv::Size(width, height), m_aiEdgeCost, labels, source, cost_curr);
				}
				if (j != height - 1) {
					depth_estimation_by_graph_cut_sWeight_add_nodes(g, nodes, cv::Size(i, j + 1), cv::Size(i, j), cv::Size(width, height), m_aiEdgeCost, labels, source, cost_curr);
				}
			}
		}
		//printf("nodes and egde set \n");

		//resolve the maximum flow/minimum cut problem
		g.maxflow();

		//update the depth labels, nodes that are still connected to the source will receive a new depth label
		for (int r = 0; r < height; ++r) {
			for (int c = 0; c < width; ++c) {
				const int pp = r * width + c;
				if (g.what_segment(nodes[pp]) != Graph::SOURCE)
					labels(r, c) = ushort(source);
			}
		}
		nodes.clear();

		std::cout << " - done" << std::endl;
		
		/*
		cv::namedWindow("labels", cv::WINDOW_NORMAL);
		cv::imshow("labels", labels);
		cv::waitKey(0);
		*/

	}

	cv::Mat depth;
	labels.convertTo(depth, CV_8U, 1.0);

	return depth;
}

/** 
 * @brief Enumerates the available depth extraction methods. 
 * Used in main() to choose between the min and graph-cut methods with a command-line argument.
*/
enum class DepthMethod {
    Min,
    GraphCut
};

/**
 * @brief Reads the command-line argument to choose the depth extraction method.
 * @param arg Command-line argument string.
 * @return DepthMethod enum value corresponding to the chosen method.
 */
DepthMethod read_method_from_arg(std::string const& arg) {
    
	// simple arg parsing, could be more robust but it does the job for now
	if (arg == "min" || arg == "m") return DepthMethod::Min;
    if (arg == "graphcut" || arg == "gc") return DepthMethod::GraphCut;

    std::cerr << "Usage: [min/m|graphcut/gc]" << std::endl;
    std::exit(EXIT_FAILURE);
	
}

/**
 * @brief Program entry point: load cameras, run CUDA planesweep, choose depth, write the image.
 * Use command-line argument "min" or "m" for the cheap planesweep-only depth extraction, "graphcut" or "gc" 
 * for the graph-cut refinement (much prettier but much slower).
 * @return EXIT_SUCCESS when everything made it to the end without drama.
 */
int main(int argc, char *argv[]) {

	// pretty wow :3
	cuda_utils::printDeviceInfo();

	// choose between the min and graph-cut depth extraction methods
	DepthMethod depth_method;
	if (argc > 1) {
		depth_method = read_method_from_arg(argv[1]);
	} else {
		printf("%s[w] No depth extraction method specified, defaulting to min.\n\n%s", EA_YELLOW, EA_DEFAULT);
		depth_method = DepthMethod::Min;
	}

	// read cams from disk and compute the depth with the chosen method
	std::vector<cam> cam_vector = read_cams("res");
	cv::Mat depth;

	if (depth_method == DepthMethod::GraphCut) {
		std::vector<cv::Mat> cost_cube = PlaneSweepKernel::sweeping_plane_cuda(cam_vector.at(0), cam_vector, 5);
		depth = depth_estimation_by_graph_cut_sWeight(cost_cube);
	} else {
		depth = PlaneSweepKernel::sweeping_plane_cuda_min_depth(cam_vector.at(0), cam_vector, 5);
	}

	printf("%s[✓] Depth estimation completed using method: %s%s\n", EA_GREEN, (depth_method == DepthMethod::GraphCut) ? "GraphCut" : "Min", EA_DEFAULT);

	// write the depth map to disk and exit
	// i removed the opencv visualization bcs on wsl it kinda breaks im not sure why
	cv::imwrite("./depth_map.png", depth);
	printf("%s[✓] Depth map written to disk as depth_map.png%s\n", EA_GREEN, EA_DEFAULT);

	return EXIT_SUCCESS;

}
