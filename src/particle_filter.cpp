#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include "particle_filter.h"
#include "map.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    //Get an intial guess using GPS noisy data

    //Set the number of particles:
    num_particles = 500;
    for (int i = 0; i < num_particles; i++) {

        random_device rd;
        default_random_engine gen(rd());
        normal_distribution<double> gps_error_x(x, std[0]);
        normal_distribution<double> gps_error_y(y, std[1]);
        normal_distribution<double> gps_error_theta(theta, std[2]);

        double particle_x = gps_error_x(gen);
        double particle_y = gps_error_y(gen);
        double particle_theta = gps_error_theta(gen);
        double weight = 1.0;

        weights.push_back(weight);

        Particle new_particle = {i, particle_x, particle_y, particle_theta, weight};
        particles.push_back(new_particle);

    }
    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // Given the time step, velocity, yaw_rate, and their std, get the position and orientation

    for (int i = 0; i < num_particles; i++) {

        if (yaw_rate == 0) {

            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        }
        else {

            particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
            particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
            particles[i].theta += yaw_rate * delta_t;
        }


        random_device rd;
        default_random_engine gen(rd());
        normal_distribution<double> pos_error_x(particles[i].x, pow(std_pos[0], 2));
        normal_distribution<double> pos_error_y(particles[i].y, pow(std_pos[1], 2));
        normal_distribution<double> pos_error_theta(particles[i].theta, pow(std_pos[2], 2));

        particles[i].x = pos_error_x(gen);
        particles[i].y = pos_error_y(gen);
        particles[i].theta = pos_error_theta(gen);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // Associate each point to the nearest neighbor 

    if (observations.size() > 0) {

        for (int i = 0; i < observations.size(); i++) {

            double temp_delta_l = 9999.0;

            for (int j = 0; j < predicted.size(); j++) {

                double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

                if (temp_delta_l > distance) {

                    temp_delta_l = distance;
                    observations[i].id = j;
                }
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], std::vector<LandmarkObs> observations, Map map_landmarks) {
    // Update weights, first transforming to the same coordinate system, and then
    // using the Multivariate-Gaussian formula

    const double denominator = sqrt(2.0 * M_PI * std_landmark[0] * std_landmark[1]);

    for (int i = 0; i < num_particles; i++) {

        double current_x = particles[i].x;
        double current_y = particles[i].y;
        double current_theta = particles[i].theta;

        vector<LandmarkObs> predicted_landmarks;
        for (int l = 0; l < map_landmarks.landmark_list.size(); l++) {

            int l_id = map_landmarks.landmark_list[l].id_i;
            double l_x = map_landmarks.landmark_list[l].x_f;
            double l_y = map_landmarks.landmark_list[l].y_f;

            double delta_x = l_x - current_x;
            double delta_y = l_y - current_y;

            double distance = dist(l_x, l_y, current_x, current_y);
            if (distance <= sensor_range) {

                l_x = delta_x * cos(current_theta) + delta_y * sin(current_theta);
                l_y = delta_y * cos(current_theta) - delta_x * sin(current_theta);
                LandmarkObs landmark_in_range = {l_id, l_x, l_y};
                predicted_landmarks.push_back(landmark_in_range);
            }
        }

        ParticleFilter::dataAssociation(predicted_landmarks, observations);

        double weight = 1.0;
        for (int obs = 0; obs < observations.size(); obs++) {

            int l_id = observations[obs].id;
            double d_x = observations[obs].x - predicted_landmarks[l_id].x;
            double d_y = observations[obs].y - predicted_landmarks[l_id].y;
            double numerator = exp(-0.5 * (pow(d_x, 2.0) * std_landmark[0] + pow(d_y, 2.0) * std_landmark[1]));
            weight *= numerator / denominator;
        }
        weights[i] = weight;
        particles[i].weight = weight;

    }
}

void ParticleFilter::resample() {
    // Resample particles with replacement with probability proportional to their weight
    // using the resampling wheel technique

    vector<Particle> new_particles;

    double r = static_cast<double> (rand()) / static_cast<double> (RAND_MAX);
    int index = r * weights.size();
    double beta = 0.0;
    double mw = 0;
    for (int ind = 0; ind < weights.size(); ind++) {

        if (weights[ind] > mw) {

            mw = weights[ind];
        }
    }
    for (int i = 0; i < weights.size(); i++) {

        r = static_cast<double> (rand()) / static_cast<double> (RAND_MAX);
        beta += r * 2.0 * mw;
        while (beta > weights[index]) {

            beta -= weights[index];
            index = (index + 1) % weights.size();
        }
        new_particles.push_back(particles[index]);
    }
    particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
    ofstream dataFile;
    dataFile.open(filename, ios::app);
    for (int i = 0; i < num_particles; ++i) {
        dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
    }
    dataFile.close();
}

